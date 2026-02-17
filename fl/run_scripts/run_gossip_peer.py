import asyncio
import argparse
import signal
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor.actor_system import ActorSystem
from actor.supervisor import Supervisor, MonitorChild
from fl.gossip_peer import GossipPeer, GossipConfig
from actor.messages import GossipPeerJoin


def load_region_data(region: str):
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "dataset",
        f"region_{region}.npz"
    )

    if not os.path.exists(dataset_path):
        print(f"[WARNING] Dataset not found: {dataset_path}")
        return None, None

    try:
        data = np.load(dataset_path)
        X = data['X']
        y = data['y']
        print(f"[INFO] Loaded region {region}: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None, None


async def main(
    peer_id: str,
    region: str,
    port: int = 0,
    peers: str = "",
    local_epochs: int = 1,
    convergence_eps: float = 0.001,
    max_rounds: int = None,
    reporter: str = "",
    health_check: bool = True,
):
    system = ActorSystem(name=f"gossip-peer-{peer_id}", host="0.0.0.0", port=port)

    try:
        await system.start_server()
        print(f"[INFO] Gossip Peer '{peer_id}' TCP server started on port {system.port}")
    except OSError as e:
        if getattr(e, "errno", None) == 10048 or "address already in use" in str(e).lower():
            print(f"[ERROR] Port {port} is already in use!")
            print(f"[ERROR] Find the process holding it:")
            print(f"        netstat -ano | findstr :{port}")
            print(f"[ERROR] Then kill it (example):")
            print(f"        taskkill /PID <PID> /F")
            print(f"[ERROR] Or run cleanup_processes.ps1 and retry.")
            sys.exit(1)
        raise

    X, y = load_region_data(region)

    reporter_host = None
    reporter_port = None
    if reporter:
        if ":" in reporter:
            reporter_host, reporter_port_str = reporter.split(":", 1)
            reporter_port = int(reporter_port_str)
        else:
            print(f"[WARNING] Reporter format should be host:port, got: {reporter}")

    seed_peers = []
    if peers:
        for peer_spec in peers.split(","):
            peer_spec = peer_spec.strip()
            if ":" in peer_spec:
                peer_host, peer_port_str = peer_spec.split(":", 1)
                seed_peers.append((peer_host, int(peer_port_str)))

    config = GossipConfig(
        peer_id=peer_id,
        fanout=2,
        gossip_interval=3.0,
        local_epochs=local_epochs,
        batch_size=32,
        lr=0.01,
        convergence_eps=convergence_eps,
        convergence_patience=3,
        max_rounds=max_rounds,
        reporter_host=reporter_host,
        reporter_port=reporter_port,
        seed_peers=seed_peers,
    )

    if health_check:
        supervisor_ref = system.actor_of(
            Supervisor,
            "supervisor",
            health_check_interval=5.0,
            health_timeout=3.0,
        )
        print("[INFO] Supervisor started (health check enabled for gossip peer)")
        supervisor_ref.tell(
            MonitorChild(
                child_id="peer",
                actor_class=GossipPeer,
                kwargs={
                    "config": config,
                    "data": (X, y) if X is not None else None,
                },
            )
        )
        while "peer" not in system._actors:
            await asyncio.sleep(0.1)
        peer_ref = system.actor_of(
            GossipPeer,
            "peer",
            config=config,
            data=(X, y) if X is not None else None,
        )
    else:
        supervisor_ref = None
        peer_ref = system.actor_of(
            GossipPeer,
            "peer",
            config=config,
            data=(X, y) if X is not None else None,
        )

    print(f"[INFO] Gossip Peer '{peer_id}' ready (no coordinator needed!)")
    print(f"[INFO] Autonomous training loop started...")
    if health_check:
        print("[INFO] Health checks ENABLED - Supervisor monitoring gossip peer")
    else:
        print("[INFO] Health checks DISABLED")

    if peers:
        print(f"[INFO] Connecting to peers: {peers}")
        for peer_spec in peers.split(","):
            peer_spec = peer_spec.strip()
            if ":" in peer_spec:
                peer_host, peer_port = peer_spec.split(":")
                peer_port = int(peer_port)
                try:
                    other_peer_ref = system.remote_ref("peer", peer_host, peer_port)

                    advertised_host = "localhost"
                    if system.host not in ("0.0.0.0", "127.0.0.1", "localhost"):
                        advertised_host = system.host

                    other_peer_ref.tell(
                        GossipPeerJoin(peer_id=peer_id, host=advertised_host, port=system.port),
                        sender=peer_ref,
                    )
                    print(f"[INFO] Sent join message to {peer_host}:{peer_port}")
                except Exception as e:
                    print(f"[WARNING] Could not connect to {peer_spec}: {e}")

    print(f"[INFO] Press Ctrl+C to stop.\n")

    stop_event = asyncio.Event()

    def signal_handler():
        print("\n[INFO] Shutdown signal received...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    if sys.platform == "win32":
        signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                pass

    await stop_event.wait()
    print(f"[INFO] Shutting down peer '{peer_id}' ...")
    await system.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("peer_id", type=str)
    parser.add_argument("region", type=str, help="A/B/C/D")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--peers", type=str, default="", help="comma-separated host:port seeds")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--convergence-eps", type=float, default=0.001)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--reporter", type=str, default="")
    parser.add_argument("--health-check", action="store_true", default=True, help="Enable health checks (default: enabled)")
    parser.add_argument("--no-health-check", action="store_false", dest="health_check", help="Disable health checks")

    args = parser.parse_args()

    asyncio.run(
        main(
            peer_id=args.peer_id,
            region=args.region,
            port=args.port,
            peers=args.peers,
            local_epochs=args.local_epochs,
            convergence_eps=args.convergence_eps,
            max_rounds=args.max_rounds,
            reporter=args.reporter,
            health_check=args.health_check,
        )
    )

import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor.actor_system import ActorSystem
from fl.worker import RegionWorker
from fl.provider import RegisterWorker


async def main(
    host: str,
    port: int,
    provider_host: str,
    provider_port: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    data_dir: str
):
    system = ActorSystem(name="workers-abc-system", host=host, port=port)
    await system.start_server()
    print(f"[INFO] Workers TCP server started on {host}:{system.port}")

    provider_ref = system.remote_ref("provider", provider_host, provider_port)
    print(f"[INFO] Created remote reference to provider at {provider_host}:{provider_port}")

    regions = ["A", "B", "C"]
    worker_ids = {}

    for region in regions:
        worker_id = f"worker-{region}"
        worker_ids[region] = worker_id

        system.actor_of(
            RegionWorker,
            worker_id,
            region=region,
            data_dir=data_dir,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr
        )

        worker_actor = system._actors[worker_id]
        worker_actor.provider_ref = provider_ref

        print(f"[INFO] Worker created: {worker_id} (region {region})")

    await asyncio.sleep(1)

    # IMPORTANT: provider must be able to call back this worker via host+port
    advertised_host = "localhost" if host in ("0.0.0.0", "127.0.0.1", "localhost") else host

    for region in regions:
        provider_ref.tell(RegisterWorker(
            worker_id=worker_ids[region],
            region=region,
            host=advertised_host,
            port=system.port
        ))
        print(f"[INFO] Registered {worker_ids[region]} with provider at {provider_host}:{provider_port}")

    print("\n[INFO] Waiting for training commands...")
    print("[INFO] Press Ctrl+C to stop\n")

    stop_event = asyncio.Event()

    def signal_handler():
        print("\n[INFO] Shutdown signal received...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass

    await stop_event.wait()
    print("[INFO] Shutting down workers system...")
    await system.shutdown()
    print("[INFO] Goodbye!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run workers A/B/C in one process (register to Provider)")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=5002, help="Workers TCP port (default: 5002)")
    p.add_argument("--provider-host", type=str, default="localhost", help="Provider host (default: localhost)")
    p.add_argument("--provider-port", type=int, default=5001, help="Provider port (default: 5001)")
    p.add_argument("--epochs", type=int, default=3, help="Local epochs per round (default: 3)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    p.add_argument("--data-dir", type=str, default="dataset", help="Dataset directory (default: dataset)")
    args = p.parse_args()

    asyncio.run(main(
        args.host, args.port,
        args.provider_host, args.provider_port,
        args.epochs, args.batch_size, args.lr,
        args.data_dir
    ))

import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor.actor_system import ActorSystem
from actor.supervisor import Supervisor, MonitorChild
from fl.worker import RegionWorker


async def main(
    host: str,
    port: int,
    provider_host: str,
    provider_port: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    data_dir: str,
    health_check: bool = True,
):
    system = ActorSystem(name="workers-abc-system", host=host, port=port)
    try:
        await system.start_server()
        print(f"[INFO] Workers TCP server started on {host}:{system.port}")
    except OSError as e:
        if e.errno == 10048 or "address already in use" in str(e).lower():
            print(f"[ERROR] Port {port} is already in use!")
            print(f"[ERROR] Kill existing Python processes or use a different port.")
            print(f"[ERROR] Run: Get-Process python | Stop-Process -Force")
            sys.exit(1)
        raise

    if health_check:
        supervisor_ref = system.actor_of(Supervisor, "supervisor", health_check_interval=5.0, health_timeout=3.0)
        print("[INFO] Supervisor started (health check enabled for workers)")
    else:
        supervisor_ref = None

    provider_ref = system.remote_ref("provider", provider_host, provider_port)
    print(f"[INFO] Created remote reference to provider at {provider_host}:{provider_port}")

    regions = ["A", "B", "C"]
    
    for region in regions:
        worker_id = f"worker-{region}"
        
        if supervisor_ref:
            supervisor_ref.tell(MonitorChild(
                child_id=worker_id,
                actor_class=RegionWorker,
                kwargs={
                    "region": region,
                    "data_dir": data_dir,
                    "local_epochs": local_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "provider_ref": provider_ref
                }
            ))
        else:
            system.actor_of(
                RegionWorker,
                worker_id,
                region=region,
                data_dir=data_dir,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                provider_ref=provider_ref
            )
    print("[INFO] Press Ctrl+C to stop\n")

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
    p.add_argument("--health-check", action="store_true", default=True, help="Enable health checks (default: enabled)")
    p.add_argument("--no-health-check", action="store_false", dest="health_check", help="Disable health checks")
    args = p.parse_args()

    asyncio.run(main(
        args.host, args.port,
        args.provider_host, args.provider_port,
        args.epochs, args.batch_size, args.lr,
        args.data_dir, args.health_check
    ))

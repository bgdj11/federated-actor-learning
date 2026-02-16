import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor.actor_system import ActorSystem
from actor.supervisor import Supervisor, MonitorChild
from fl.aggregator import Aggregator


async def main(port: int, provider_host: str, provider_port: int, health_check: bool = True):
    system = ActorSystem(name="aggregator-system", host="0.0.0.0", port=port)
    try:
        await system.start_server()
        print(f"[INFO] Aggregator TCP server started on port {system.port}")
    except OSError as e:
        if e.errno == 10048 or "address already in use" in str(e).lower():
            print(f"[ERROR] Port {port} is already in use!")
            print(f"[ERROR] Kill existing Python processes or use a different port.")
            print(f"[ERROR] Run: Get-Process python | Stop-Process -Force")
            sys.exit(1)
        raise

    if health_check:
        supervisor_ref = system.actor_of(Supervisor, "supervisor", health_check_interval=5.0, health_timeout=3.0)
        print("[INFO] Supervisor started (health check enabled for aggregator)")
    else:
        supervisor_ref = None

    provider_ref = system.remote_ref("provider", provider_host, provider_port)
    print(f"[INFO] Created remote reference to provider at {provider_host}:{provider_port}")

    if supervisor_ref:
        supervisor_ref.tell(MonitorChild(
            child_id="aggregator",
            actor_class=Aggregator,
            kwargs={"provider_ref": provider_ref}
        ))
    else:
        system.actor_of(Aggregator, "aggregator", provider_ref=provider_ref)
    
    print("[INFO] Aggregator actor ready. Press Ctrl+C to stop.\n")

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
    print("[INFO] Shutting down aggregator system...")
    await system.shutdown()
    print("[INFO] Goodbye!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Aggregator (aggregation-only service)")
    p.add_argument("--port", type=int, default=5000, help="Aggregator TCP port (default: 5000)")
    p.add_argument("--provider-host", type=str, default="localhost", help="Provider host (default: localhost)")
    p.add_argument("--provider-port", type=int, default=5001, help="Provider port (default: 5001)")
    p.add_argument("--health-check", action="store_true", default=True, help="Enable health checks (default: enabled)")
    p.add_argument("--no-health-check", action="store_false", dest="health_check", help="Disable health checks")
    args = p.parse_args()

    asyncio.run(main(args.port, args.provider_host, args.provider_port, args.health_check))

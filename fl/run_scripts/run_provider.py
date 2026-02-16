import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor.actor_system import ActorSystem
from actor.supervisor import Supervisor, MonitorChild
from fl.provider import Provider

async def main(
    port: int,
    rounds: int,
    workers: int,
    mu: float,
    health_check: bool = True,
):
    system = ActorSystem(name="provider-system", host="0.0.0.0", port=port)
    try:
        await system.start_server()
        print(f"[INFO] Provider TCP server started on port {system.port}")
    except OSError as e:
        if e.errno == 10048 or "address already in use" in str(e).lower():
            print(f"[ERROR] Port {port} is already in use!")
            print(f"[ERROR] Kill existing Python processes or use a different port.")
            print(f"[ERROR] Run: Get-Process python | Stop-Process -Force")
            sys.exit(1)
        raise

    if health_check:
        supervisor_ref = system.actor_of(Supervisor, "supervisor", health_check_interval=5.0, health_timeout=3.0)
        print("[INFO] Supervisor started (health check enabled)")
        
        supervisor_ref.tell(MonitorChild(
            child_id="provider",
            actor_class=Provider,
            kwargs={
                "num_workers": workers,
                "num_rounds": rounds,
                "auto_start": True,
                "mu": mu
            }
        ))
        
        while "provider" not in system._actors:
            await asyncio.sleep(0.1)
    else:
        supervisor_ref = None
        system.actor_of(
            Provider,
            "provider",
            num_workers=workers,
            num_rounds=rounds,
            auto_start=True,
            mu=mu
        )
    
    print("[INFO] Provider ready. Waiting for aggregator, workers, and evaluator...")
    if health_check:
        print("[INFO] Health checks ENABLED - Supervisor monitoring components")
    else:
        print("[INFO] Health checks DISABLED")
    print("[INFO] Waiting for aggregator, workers, and evaluator to register...")
    print("[INFO] Press Ctrl+C to stop.\n")

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
    print("[INFO] Shutting down provider system...")
    await system.shutdown()
    print("[INFO] Goodbye!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Provider (scheduler/orchestrator)")
    p.add_argument("--port", type=int, default=5001, help="Provider TCP port (default: 5001)")
    p.add_argument("--rounds", type=int, default=5, help="Number of FL rounds (default: 5)")
    p.add_argument("--workers", type=int, default=3, help="Expected number of workers (default: 3)")
    p.add_argument("--mu", type=float, default=0.0, help="FedProx mu (default: 0.0)")
    p.add_argument("--health-check", action="store_true", default=True, help="Enable health checks (default: enabled)")
    p.add_argument("--no-health-check", action="store_false", dest="health_check", help="Disable health checks")
    args = p.parse_args()

    asyncio.run(main(args.port, args.rounds, args.workers, args.mu, args.health_check))

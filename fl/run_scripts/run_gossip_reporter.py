import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor.actor_system import ActorSystem
from fl.gossip_reporter import GossipReporter, GossipReporterConfig


async def main(port: int, report_interval: float, startup_delay: float):
    system = ActorSystem(name="gossip-reporter-system", host="0.0.0.0", port=port)
    try:
        await system.start_server()
        print(f"[INFO] Gossip Reporter TCP server started on port {system.port}")
    except OSError as e:
        if e.errno == 10048 or "address already in use" in str(e).lower():
            print(f"[ERROR] Port {port} is already in use!")
            print(f"[ERROR] Run: Get-Process python | Stop-Process -Force")
            sys.exit(1)
        raise

    config = GossipReporterConfig(
        reporter_id="reporter-1",
        eval_interval=5,
        startup_delay_sec=startup_delay,
        report_interval_sec=report_interval,
        gossip_log_every=10,
    )
    system.actor_of(GossipReporter, "reporter", config=config)

    print("[INFO] Gossip Reporter ready.")
    print("[INFO] Reporter is optional - peers run even without it.")
    print("[INFO] Reporter only monitors and logs progress.")
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
    print("[INFO] Shutting down reporter system...")
    await system.shutdown()
    print("[INFO] Goodbye!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Gossip Reporter (optional monitoring)")
    p.add_argument("--port", type=int, default=5200, help="Reporter TCP port (default: 5200)")
    p.add_argument(
        "--report-interval",
        type=float,
        default=5.0,
        help="Reporter summary interval in seconds (default: 5.0)",
    )
    p.add_argument(
        "--startup-delay",
        type=float,
        default=2.0,
        help="Initial delay before first reporting loop in seconds (default: 2.0)",
    )
    args = p.parse_args()

    asyncio.run(main(args.port, args.report_interval, args.startup_delay))

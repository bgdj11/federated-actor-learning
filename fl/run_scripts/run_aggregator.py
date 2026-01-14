import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor.actor_system import ActorSystem
from fl.aggregator import Aggregator


async def main(port: int):
    system = ActorSystem(name="aggregator-system", host="0.0.0.0", port=port)
    await system.start_server()
    print(f"[INFO] Aggregator TCP server started on port {system.port}")

    system.actor_of(Aggregator, "aggregator")
    print("[INFO] Aggregator actor ready. Press Ctrl+C to stop.\n")

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
    print("[INFO] Shutting down aggregator system...")
    await system.shutdown()
    print("[INFO] Goodbye!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Aggregator (aggregation-only service)")
    p.add_argument("--port", type=int, default=5000, help="Aggregator TCP port (default: 5000)")
    args = p.parse_args()

    asyncio.run(main(args.port))

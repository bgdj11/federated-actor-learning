import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor.actor_system import ActorSystem
from fl.provider import Provider
from fl.aggregator import RegisterProvider

async def main(
    port: int,
    agg_host: str,
    agg_port: int,
    rounds: int,
    workers: int,
    mu: float,
):
    system = ActorSystem(name="provider-system", host="0.0.0.0", port=port)
    await system.start_server()
    print(f"[INFO] Provider TCP server started on port {system.port}")

    system.actor_of(
        Provider,
        "provider",
        num_workers=workers,
        num_rounds=rounds,
        auto_start=True,
        mu=mu
    )

    provider_actor = system._actors["provider"]

    aggregator_ref = system.remote_ref("aggregator", agg_host, agg_port)
    provider_actor.set_aggregator_ref(aggregator_ref)
    aggregator_ref.tell(RegisterProvider(
        provider_id="provider",
        host="localhost",
        port=system.port
    ))
    print(f"[INFO] Registered provider with aggregator (provider listening on {system.port})")


    print(f"[INFO] Provider ready. Using Aggregator at {agg_host}:{agg_port}")
    print("[INFO] Waiting for workers + evaluator to register...")
    print("[INFO] Press Ctrl+C to stop.\n")

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
    print("[INFO] Shutting down provider system...")
    await system.shutdown()
    print("[INFO] Goodbye!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Provider (scheduler/orchestrator)")
    p.add_argument("--port", type=int, default=5001, help="Provider TCP port (default: 5001)")
    p.add_argument("--agg-host", type=str, default="localhost", help="Aggregator host (default: localhost)")
    p.add_argument("--agg-port", type=int, default=5000, help="Aggregator port (default: 5000)")
    p.add_argument("--rounds", type=int, default=5, help="Number of FL rounds (default: 5)")
    p.add_argument("--workers", type=int, default=3, help="Expected number of workers (default: 3)")
    p.add_argument("--mu", type=float, default=0.0, help="FedProx mu (default: 0.0)")
    args = p.parse_args()

    asyncio.run(main(args.port, args.agg_host, args.agg_port, args.rounds, args.workers, args.mu))

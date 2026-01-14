import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor.actor_system import ActorSystem
from fl.evaluator import Evaluator
from fl.provider import RegisterEvaluator


async def main(provider_host: str, provider_port: int, data_dir: str, eval_port: int):
    system = ActorSystem(name="evaluator-system", host="0.0.0.0", port=eval_port)
    await system.start_server()
    print(f"[INFO] Evaluator TCP server started on port {system.port}")

    provider_ref = system.remote_ref("provider", provider_host, provider_port)
    print(f"[INFO] Created remote reference to provider at {provider_host}:{provider_port}")

    evaluator_ref = system.actor_of(Evaluator, "evaluator", data_dir=data_dir)
    evaluator_actor = system._actors["evaluator"]

    # evaluator reports to provider
    evaluator_actor.provider_ref = provider_ref

    await asyncio.sleep(1)

    provider_ref.tell(RegisterEvaluator(
        evaluator_id="evaluator",
        host="localhost",
        port=system.port
    ))
    print(f"[INFO] Registered evaluator with provider (listening on {system.port})")
    print("[INFO] Waiting for models... Ctrl+C to stop.\n")

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
    print("[INFO] Shutting down evaluator system...")
    await system.shutdown()
    print("[INFO] Goodbye!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Evaluator (registers to Provider)")
    p.add_argument("--provider-host", type=str, default="localhost", help="Provider host (default: localhost)")
    p.add_argument("--provider-port", type=int, default=5001, help="Provider port (default: 5001)")
    p.add_argument("--data-dir", type=str, default="dataset", help="Dataset dir (default: dataset)")
    p.add_argument("--eval-port", type=int, default=5010, help="Evaluator TCP port (default: 5010)")
    args = p.parse_args()

    asyncio.run(main(args.provider_host, args.provider_port, args.data_dir, args.eval_port))

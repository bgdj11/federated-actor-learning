import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import ActorSystem
from fl.worker import RegionWorker
from fl.aggregator import RegisterWorker


async def main(
    host: str,
    port: int,
    agg_host: str,
    agg_port: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    data_dir: str
):

    system = ActorSystem(name="workers-abc-system", host=host, port=port)
    await system.start_server()
    print(f"[INFO] TCP server started on {host}:{system.port}")

    aggregator_ref = system.remote_ref("aggregator", agg_host, agg_port)
    print(f"[INFO] Aggregator ref: {agg_host}:{agg_port}")

    regions = ["A", "B", "C"]
    worker_ids = {}

    for region in regions:
        worker_id = f"worker-{region}"
        worker_ids[region] = worker_id

        worker_ref = system.actor_of(
            RegionWorker,
            worker_id,
            region=region,
            data_dir=data_dir,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr
        )

        worker_actor = system._actors[worker_id]
        worker_actor.aggregator_ref = aggregator_ref

        print(f"[INFO] Worker {region} created with id={worker_id}")

    await asyncio.sleep(1)

    for region in regions:
        worker_id = worker_ids[region]
        register_msg = RegisterWorker(
            worker_id=worker_id,
            region=region,
            host="localhost" if host in ("0.0.0.0", "127.0.0.1", "localhost") else host,
            port=system.port
        )
        aggregator_ref.tell(register_msg)
        print(f"[INFO] Registered {worker_id} (region {region}) with aggregator")

    print("[INFO] Waiting for training commands...")
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

    try:
        while not stop_event.is_set():
            await asyncio.sleep(1)
    finally:
        print("[INFO] Shutting down workers system...")
        await system.shutdown()
        print("[INFO] Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3 FL Workers (A/B/C) in one process")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host for workers TCP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5001,
                        help="Port for workers TCP server (default: 5001)")
    parser.add_argument("--agg-host", type=str, default="localhost",
                        help="Aggregator host (default: localhost)")
    parser.add_argument("--agg-port", type=int, default=5000,
                        help="Aggregator port (default: 5000)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Local epochs per round (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--data-dir", type=str, default="dataset",
                        help="Dataset directory (default: dataset)")

    args = parser.parse_args()

    try:
        asyncio.run(main(
            args.host, args.port,
            args.agg_host, args.agg_port,
            args.epochs, args.batch_size, args.lr,
            args.data_dir
        ))
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

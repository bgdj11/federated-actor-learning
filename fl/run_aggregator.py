import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import ActorSystem
from fl.aggregator import Aggregator, RegisterWorker, RegisterEvaluator


async def main(port: int, num_rounds: int, num_workers: int, mu: float):
   
    system = ActorSystem(name="aggregator-system", host="0.0.0.0", port=port)

    await system.start_server()
    print(f"[INFO] TCP Server started on port {system.port}")
    
    aggregator = system.actor_of(
        Aggregator, 
        "aggregator",
        num_workers=num_workers,
        num_rounds=num_rounds,
        auto_start=True,
        mu=mu
    )
    
    print(f"[INFO] Aggregator actor created: {aggregator.actor_id}")
    print(f"[INFO] Waiting for {num_workers} workers to connect...")
    print(f"[INFO] Press Ctrl+C to stop\n")
    
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
        agg_actor = system._actors.get("aggregator")
        
        while not stop_event.is_set():
            await asyncio.sleep(1)
            
            if agg_actor and agg_actor.training_complete:
                print("\n[INFO] Training complete! Keeping server alive for 10 more seconds...")
                await asyncio.sleep(10)
                break
                
    except asyncio.CancelledError:
        pass
    finally:
        print("[INFO] Shutting down aggregator system...")
        await system.shutdown()
        print("[INFO] Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FL Aggregator Server")
    parser.add_argument("--port", type=int, default=5000, help="TCP port (default: 5000)")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds (default: 5)")
    parser.add_argument("--workers", type=int, default=3, help="Expected number of workers (default: 3)")
    parser.add_argument("--mu", type=float, default=0.0, help="FedProx mu parameter (default: 0.0)")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.port, args.rounds, args.workers, args.mu))
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

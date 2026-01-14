import asyncio
import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import ActorSystem
from fl.evaluator import Evaluator
from fl.aggregator import RegisterEvaluator


async def main(host: str, port: int, data_dir: str, eval_port: int):
   
    system = ActorSystem(name="evaluator-system", host="0.0.0.0", port=eval_port)
    
    await system.start_server()
    print(f"[INFO] Evaluator TCP server started on port {system.port}")
    
    aggregator_ref = system.remote_ref("aggregator", host, port)
    print(f"[INFO] Created remote reference to aggregator at {host}:{port}")
    
    evaluator_ref = system.actor_of(
        Evaluator,
        "evaluator",
        data_dir=data_dir
    )
    
    evaluator_actor = system._actors["evaluator"]
    evaluator_actor.aggregator_ref = aggregator_ref
    
    print(f"[INFO] Created evaluator actor: {evaluator_ref.actor_id}")

    await asyncio.sleep(2)
    
    register_msg = RegisterEvaluator(
        evaluator_id="evaluator",
        host="localhost",
        port=system.port
    )
    aggregator_ref.tell(register_msg)
    print(f"[INFO] Registered with aggregator (listening on port {system.port})")
    print("\n[INFO] Waiting for models to evaluate...")
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
            
    except asyncio.CancelledError:
        pass
    finally:
        print("[INFO] Shutting down evaluator...")
        await system.shutdown()
        print("[INFO] Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FL Evaluator")
    parser.add_argument("--host", type=str, default="localhost", help="Aggregator host (default: localhost)")
    parser.add_argument("--port", type=int, default=5000, help="Aggregator port (default: 5000)")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset directory (default: dataset)")
    parser.add_argument("--eval-port", type=int, default=5010, help="Evaluator listen port (default: 5010)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.host, args.port, args.data_dir, args.eval_port))
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

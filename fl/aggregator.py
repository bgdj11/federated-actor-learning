import numpy as np
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import Actor, ActorRef
from actor.messages import (
    Message, TrainRequest, ModelUpdate, GlobalModelBroadcast,
    StartRound, HealthPing, HealthAck, Shutdown
)
from fl.model import SimpleClassifier, federated_averaging


@dataclass
class RegisterWorker(Message):
    worker_id: str = ""
    region: str = ""
    host: str = "localhost"
    port: int = 0

@dataclass
class RegisterEvaluator(Message):
    evaluator_id: str = ""
    host: str = "localhost"
    port: int = 0

@dataclass
class EvaluationResult(Message):
    round_idx: int = 0
    accuracy: float = 0.0
    loss: float = 0.0


class Aggregator(Actor):
    
    def __init__(self, num_workers: int = 3, num_rounds: int = 5, 
                 auto_start: bool = True, mu: float = 0.0):

        super().__init__()
        self.num_workers = num_workers
        self.num_rounds = num_rounds
        self.auto_start = auto_start
        self.mu = mu
        self.global_model: SimpleClassifier = None
        
        self.workers: dict[str, ActorRef] = {}
        self.worker_regions: dict[str, str] = {}  # worker_id -> region
        
        self.evaluator_ref: Optional[ActorRef] = None
        
        self.current_round = 0
        
        self.round_updates: list[tuple[dict, int]] = []  # (weights, num_samples)
        self.round_metrics: list[dict] = []

        self.history: list[dict] = []

        self.training_started = False
        self.training_complete = False
        
    async def pre_start(self):
        self.global_model = SimpleClassifier()
        self.log.info(f"Aggregator started. Waiting for {self.num_workers} workers...")
        
    async def receive(self, msg: Message):
        if isinstance(msg, RegisterWorker):
            await self._handle_register_worker(msg)
            
        elif isinstance(msg, RegisterEvaluator):
            await self._handle_register_evaluator(msg)
            
        elif isinstance(msg, ModelUpdate):
            await self._handle_model_update(msg)
            
        elif isinstance(msg, EvaluationResult):
            await self._handle_evaluation_result(msg)
            
        elif isinstance(msg, StartRound):
            await self._start_round(msg.round_idx)
            
        elif isinstance(msg, HealthPing):
            self.context.self_ref.tell(HealthAck(
                actor_id=self.actor_id,
                status="alive"
            ))
            
        elif isinstance(msg, Shutdown):
            self.log.info("Shutting down aggregator...")
            self._print_summary()
            
    async def _handle_register_worker(self, msg: RegisterWorker):
        worker_id = msg.worker_id
        
        if worker_id in self.workers:
            self.log.warning(f"Worker {worker_id} already registered!")
            return
            
        worker_ref = self.context._system.remote_ref(worker_id, msg.host, msg.port)
        self.workers[worker_id] = worker_ref
        self.worker_regions[worker_id] = msg.region
        
        self.log.info(f"Worker registered: {worker_id} (region {msg.region}) at {msg.host}:{msg.port}. "
                      f"Total: {len(self.workers)}/{self.num_workers}")
        
        # check if we can start training
        if len(self.workers) >= self.num_workers and self.auto_start:
            if not self.training_started:
                self.log.info("All workers registered! Starting training...")
                await asyncio.sleep(1)
                await self._start_round(1)
                
    async def _handle_register_evaluator(self, msg: RegisterEvaluator):
        self.log.info(f"Evaluator registered: {msg.evaluator_id} at {msg.host}:{msg.port}")
        self.evaluator_ref = self.context._system.remote_ref(msg.evaluator_id, msg.host, msg.port)
        
    async def _start_round(self, round_idx: int):
        if self.training_complete:
            self.log.warning("Training already complete!")
            return
            
        self.current_round = round_idx
        self.round_updates = []
        self.round_metrics = []
        self.training_started = True
        
        self.log.info(f"\n{'='*50}")
        self.log.info(f"ROUND {self.current_round}/{self.num_rounds}")
        self.log.info(f"{'='*50}")
        
        train_request = TrainRequest(
            round_idx=self.current_round,
            global_weights=self.global_model.get_weights(),
            mu=self.mu
        )
        
        for worker_id, worker_ref in self.workers.items():
            if worker_ref:
                worker_ref.tell(train_request)
                self.log.info(f"Sent TrainRequest to {worker_id}")
            else:
                self.log.warning(f"No reference for worker {worker_id}")
                
    async def _handle_model_update(self, msg: ModelUpdate):
        self.log.info(f"Received ModelUpdate from {msg.worker_id} "
                      f"({msg.num_samples} samples, "
                      f"acc={msg.metrics.get('accuracy', 0):.4f})")
        
        self.round_updates.append((msg.weights, msg.num_samples))
        self.round_metrics.append(msg.metrics)
        
        if len(self.round_updates) >= self.num_workers:
            await self._aggregate_and_broadcast()
            
    async def _aggregate_and_broadcast(self):
        self.log.info(f"\nAggregating {len(self.round_updates)} updates...")
        
        aggregated_weights = federated_averaging(self.round_updates)
        self.global_model.set_weights(aggregated_weights)
        
        avg_loss = np.mean([m['loss'] for m in self.round_metrics])
        avg_acc = np.mean([m['accuracy'] for m in self.round_metrics])
        
        self.log.info(f"Round {self.current_round} aggregated: "
                      f"avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}")
        
        self.history.append({
            'round': self.current_round,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_acc,
            'worker_metrics': self.round_metrics.copy()
        })
        
        broadcast_msg = GlobalModelBroadcast(
            round_idx=self.current_round,
            weights=self.global_model.get_weights()
        )
        
        for worker_id, worker_ref in self.workers.items():
            if worker_ref:
                worker_ref.tell(broadcast_msg)
                
        if self.evaluator_ref:
            self.evaluator_ref.tell(broadcast_msg)
            self.log.info("Sent model to evaluator for evaluation")
        else:
            await self._proceed_to_next_round()
            
    async def _handle_evaluation_result(self, msg: EvaluationResult):
        self.log.info(f"Evaluation result for round {msg.round_idx}: "
                      f"accuracy={msg.accuracy:.4f}, loss={msg.loss:.4f}")
        
        if self.history and self.history[-1]['round'] == msg.round_idx:
            self.history[-1]['eval_accuracy'] = msg.accuracy
            self.history[-1]['eval_loss'] = msg.loss
            
        await self._proceed_to_next_round()
        
    async def _proceed_to_next_round(self):
        if self.current_round >= self.num_rounds:
            self.training_complete = True
            self.log.info(f"\n{'='*50}")
            self.log.info("TRAINING COMPLETE!")
            self.log.info(f"{'='*50}")
            self._print_summary()
        else:
            await asyncio.sleep(0.5)
            await self._start_round(self.current_round + 1)
            
    def _print_summary(self):
        self.log.info("\n" + "="*50)
        self.log.info("TRAINING SUMMARY")
        self.log.info("="*50)
        
        for h in self.history:
            eval_info = ""
            if 'eval_accuracy' in h:
                eval_info = f", eval_acc={h['eval_accuracy']:.4f}"
            self.log.info(f"Round {h['round']}: "
                          f"avg_loss={h['avg_loss']:.4f}, "
                          f"avg_acc={h['avg_accuracy']:.4f}{eval_info}")
                          
        if self.history:
            final = self.history[-1]
            self.log.info(f"\nFinal: avg_acc={final['avg_accuracy']:.4f}")
            if 'eval_accuracy' in final:
                self.log.info(f"Final eval accuracy: {final['eval_accuracy']:.4f}")
                
    async def post_stop(self):
        self.log.info("Aggregator stopped.")
        
    def set_worker_ref(self, worker_id: str, ref: ActorRef):
        if worker_id in self.workers:
            self.workers[worker_id] = ref
            self.log.info(f"Set reference for worker {worker_id}")
            
    def set_evaluator_ref(self, ref: ActorRef):
        self.evaluator_ref = ref
        self.log.info("Set evaluator reference")

import os
import sys
import asyncio
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import Actor, ActorRef
from actor.messages import (
    Message, TrainRequest, ModelUpdate, GlobalModelBroadcast,
    HealthPing, HealthAck, Shutdown
)
from fl.model import SimpleClassifier
from fl.aggregator import AggregateRound, AggregatedResult, RegisterAggregator


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


class Provider(Actor):

    def __init__(self, num_workers: int = 3, num_rounds: int = 5,
                 auto_start: bool = True, mu: float = 0.0):
        super().__init__()
        self.num_workers = num_workers
        self.num_rounds = num_rounds
        self.auto_start = auto_start
        self.mu = mu

        self.global_model: Optional[SimpleClassifier] = None

        self.workers: dict[str, ActorRef] = {}
        self.evaluator_ref: Optional[ActorRef] = None
        self.aggregator_ref: Optional[ActorRef] = None
        self._aggregator_id: Optional[str] = None
        self._evaluator_id: Optional[str] = None
        self._last_health_ack: dict[str, float] = {}
        self._health_timeout = 2.5

        self.current_round = 0
        self.training_started = False
        self.training_complete = False

        self._round_weight_updates: list[tuple[dict, int]] = []
        self._round_train_metrics: list[dict] = []

        self.history: list[dict] = []

        self._awaiting_aggregation = False
        self._health_check_task: Optional[asyncio.Task] = None

        self._pending_aggregate: Optional[AggregateRound] = None
        self._pending_eval_broadcast: Optional[GlobalModelBroadcast] = None

    async def pre_start(self):
        self.global_model = SimpleClassifier()
        self.log.info(f"Provider started. Waiting for {self.num_workers} workers...")
        self._health_check_task = asyncio.create_task(self._periodic_health_check())

    async def _periodic_health_check(self):
        await asyncio.sleep(8.0)
        
        while True:
            try:
                await asyncio.sleep(2.0)

                if self.aggregator_ref:
                    sent_at = asyncio.get_event_loop().time()
                    self.aggregator_ref.tell(HealthPing(), sender=self.context.self_ref)
                    asyncio.create_task(self._check_health_timeout("aggregator", sent_at))

                if self.evaluator_ref:
                    sent_at = asyncio.get_event_loop().time()
                    self.evaluator_ref.tell(HealthPing(), sender=self.context.self_ref)
                    asyncio.create_task(self._check_health_timeout("evaluator", sent_at))

            except Exception as e:
                self.log.error(f"Health check error: {e}")

    async def _check_health_timeout(self, target: str, sent_at: float):
        await asyncio.sleep(self._health_timeout)
        last_ack = self._last_health_ack.get(target)

        if last_ack is not None and last_ack <= sent_at:
            if target == "aggregator":
                if self.aggregator_ref is not None:
                    self.log.warning("Aggregator health timeout; waiting for re-registration")
                    self.aggregator_ref = None
            elif target == "evaluator":
                if self.evaluator_ref is not None:
                    self.log.warning("Evaluator health timeout; waiting for re-registration")
                    self.evaluator_ref = None

    async def receive(self, msg: Message):
        if isinstance(msg, RegisterWorker):
            await self._handle_register_worker(msg)

        elif isinstance(msg, RegisterEvaluator):
            await self._handle_register_evaluator(msg)

        elif isinstance(msg, RegisterAggregator):
            await self._handle_register_aggregator(msg)

        elif isinstance(msg, ModelUpdate):
            await self._handle_model_update(msg)

        elif isinstance(msg, AggregatedResult):
            await self._handle_aggregated_result(msg)

        elif isinstance(msg, EvaluationResult):
            await self._handle_evaluation_result(msg)

        elif isinstance(msg, HealthPing):
            if msg.sender:
                msg.sender.tell(HealthAck(actor_id=self.actor_id, status="alive"))
            else:
                self.log.warning("HealthPing received but no sender!")

        elif isinstance(msg, HealthAck):
            now = asyncio.get_event_loop().time()
            if msg.actor_id == self._aggregator_id:
                self._last_health_ack["aggregator"] = now
            elif msg.actor_id == self._evaluator_id:
                self._last_health_ack["evaluator"] = now

        elif isinstance(msg, Shutdown):
            self.log.info("Shutting down provider...")
            self._print_summary()

    async def _handle_register_worker(self, msg: RegisterWorker):
        if msg.worker_id in self.workers:
            self.log.warning(f"Worker {msg.worker_id} already registered!")
            return

        self.workers[msg.worker_id] = self.context._system.remote_ref(
            msg.worker_id, msg.host, msg.port
        )

        self.log.info(f"Worker registered: {msg.worker_id} (region {msg.region}) at {msg.host}:{msg.port}. "
                      f"Total: {len(self.workers)}/{self.num_workers}")

        if len(self.workers) >= self.num_workers and self.auto_start and not self.training_started:
            self.log.info("All workers registered! Starting training...")
            await asyncio.sleep(1)
            await self._start_round(1)

    async def _handle_register_evaluator(self, msg: RegisterEvaluator):
        self.log.info(f"Evaluator registered: {msg.evaluator_id}")
        self.evaluator_ref = self.context._system.remote_ref(
            msg.evaluator_id, msg.host, msg.port
        )
        self._evaluator_id = msg.evaluator_id
        self.log.info(f"Evaluator registration complete at {msg.host}:{msg.port}")

        if self._pending_eval_broadcast and self.evaluator_ref:
            self.evaluator_ref.tell(self._pending_eval_broadcast)
            self.log.info("Sent pending model to evaluator")
            self._pending_eval_broadcast = None

    async def _handle_register_aggregator(self, msg: RegisterAggregator):
        self.log.info(f"Aggregator registered: {msg.aggregator_id}")
        self.aggregator_ref = self.context._system.remote_ref(
            msg.aggregator_id, msg.host, msg.port
        )
        self._aggregator_id = msg.aggregator_id
        self.log.info(f"Aggregator registration complete at {msg.host}:{msg.port}")

        if self._pending_aggregate:
            self.log.info(f"Sending pending AggregateRound for round {self._pending_aggregate.round_idx}")
            self.aggregator_ref.tell(self._pending_aggregate)
            self._pending_aggregate = None
            self._awaiting_aggregation = False  # Reset flag so system can continue

    async def _start_round(self, round_idx: int):
        if self.training_complete:
            return
        if self.global_model is None:
            self.log.error("Global model not initialized!")
            return

        self.training_started = True
        self.current_round = round_idx
        self._awaiting_aggregation = False
        self._round_weight_updates = []
        self._round_train_metrics = []

        self.log.info(f"\n{'='*50}")
        self.log.info(f"ROUND {self.current_round}/{self.num_rounds}")
        self.log.info(f"{'='*50}")

        req = TrainRequest(
            round_idx=self.current_round,
            global_weights=self.global_model.get_weights(),
            mu=self.mu
        )

        for wid, wref in self.workers.items():
            wref.tell(req)
            self.log.info(f"Sent TrainRequest to {wid}")

    async def _handle_model_update(self, msg: ModelUpdate):
        self.log.info(f"Received ModelUpdate from {msg.worker_id} ({msg.num_samples} samples)")

        self._round_weight_updates.append((msg.weights, msg.num_samples))
        self._round_train_metrics.append(msg.metrics)

        if len(self._round_weight_updates) >= self.num_workers and not self._awaiting_aggregation:
            aggregate_msg = AggregateRound(
                round_idx=self.current_round,
                weight_updates=self._round_weight_updates,
                train_metrics=self._round_train_metrics
            )
            
            if not self.aggregator_ref:
                self.log.warning("No aggregator_ref set; waiting for registration")
                self._pending_aggregate = aggregate_msg
                return

            self._awaiting_aggregation = True
            self.log.info("All updates received. Sending to Aggregator...")

            self._pending_aggregate = aggregate_msg
            self.aggregator_ref.tell(aggregate_msg)

    async def _handle_aggregated_result(self, msg: AggregatedResult):
        if msg.round_idx != self.current_round:
            self.log.warning(f"Ignoring AggregatedResult for round {msg.round_idx} (current {self.current_round})")
            return
        if self.global_model is None:
            self.log.error("Global model not initialized!")
            return

        self._pending_aggregate = None

        self.global_model.set_weights(msg.weights)

        train_avg_loss = msg.train_summary.get("train_avg_loss", 0.0)
        train_avg_acc = msg.train_summary.get("train_avg_accuracy", 0.0)

        self.log.info(f"Round {self.current_round} aggregated (train): "
                      f"avg_loss={train_avg_loss:.4f}, avg_acc={train_avg_acc:.4f}")

        self.history.append({
            "round": self.current_round,
            "train_avg_loss": float(train_avg_loss),
            "train_avg_accuracy": float(train_avg_acc),
        })

        bcast = GlobalModelBroadcast(round_idx=self.current_round, weights=self.global_model.get_weights())
        for _, wref in self.workers.items():
            wref.tell(bcast)

        self._pending_eval_broadcast = bcast
        
        if self.evaluator_ref:
            self.evaluator_ref.tell(bcast)
            self.log.info("Sent model to evaluator for evaluation")
        else:
            self.log.warning("No evaluator_ref set; waiting for registration")

    async def _handle_evaluation_result(self, msg: EvaluationResult):
        self.log.info(f"Eval for round {msg.round_idx}: acc={msg.accuracy:.4f}, loss={msg.loss:.4f}")

        self._pending_eval_broadcast = None

        if self.history and self.history[-1]["round"] == msg.round_idx:
            self.history[-1]["eval_accuracy"] = float(msg.accuracy)
            self.history[-1]["eval_loss"] = float(msg.loss)

        await self._proceed_to_next_round()

    async def _proceed_to_next_round(self):
        if self.current_round >= self.num_rounds:
            self.training_complete = True
            self.log.info(f"\n{'='*50}\nTRAINING COMPLETE!\n{'='*50}")
            self._print_summary()
            return

        await asyncio.sleep(0.5)
        await self._start_round(self.current_round + 1)

    def _print_summary(self):
        if not self.history:
            self.log.info("No rounds completed.")
            return

        self.log.info("\n" + "="*50)
        self.log.info("TRAINING SUMMARY (Provider)")
        self.log.info("="*50)

        for h in self.history:
            eval_part = ""
            if "eval_accuracy" in h:
                eval_part = f", eval_acc={h['eval_accuracy']:.4f}"
            self.log.info(
                f"Round {h['round']}: train_acc={h['train_avg_accuracy']:.4f}, "
                f"train_loss={h['train_avg_loss']:.4f}{eval_part}"
            )

    async def post_stop(self):
        if self._health_check_task:
            self._health_check_task.cancel()
        self._print_summary()
        self.log.info("Provider stopped.")

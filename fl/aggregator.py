import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import Actor, ActorRef
from actor.messages import Message
from fl.model import federated_averaging


@dataclass
class AggregateRound(Message):
    round_idx: int = 0
    weight_updates: List[Tuple[dict, int]] = None
    train_metrics: List[dict] = None


@dataclass
class AggregatedResult(Message):
    round_idx: int = 0
    weights: dict = None
    train_summary: dict = None

@dataclass
class RegisterProvider(Message):
    provider_id: str = "provider"
    host: str = "localhost"
    port: int = 0

class Aggregator(Actor):
    def __init__(self):
        super().__init__()
        self.provider_ref: Optional[ActorRef] = None

    async def receive(self, msg: Message):
        if isinstance(msg, RegisterProvider):
            self.provider_ref = self.context._system.remote_ref(msg.provider_id, msg.host, msg.port)
            self.log.info(f"Registered provider at {msg.host}:{msg.port} (id={msg.provider_id})")
            return

        if isinstance(msg, AggregateRound):
            if not msg.weight_updates:
                self.log.warning("AggregateRound received with no updates!")
                return

            aggregated = federated_averaging(msg.weight_updates)

            losses = [m.get("loss", 0.0) for m in (msg.train_metrics or [])]
            accs = [m.get("accuracy", 0.0) for m in (msg.train_metrics or [])]

            train_avg_loss = float(np.mean(losses)) if losses else 0.0
            train_avg_acc = float(np.mean(accs)) if accs else 0.0

            result = AggregatedResult(
                round_idx=msg.round_idx,
                weights=aggregated,
                train_summary={
                    "train_avg_loss": train_avg_loss,
                    "train_avg_accuracy": train_avg_acc
                }
            )

            self.provider_ref.tell(result)
            self.log.info(f"Aggregated round {msg.round_idx}: "
                          f"train_avg_loss={train_avg_loss:.4f}, "
                          f"train_avg_accuracy={train_avg_acc:.4f}")
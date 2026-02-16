import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import Actor, ActorRef
from actor.messages import Message, HealthPing, HealthAck
from fl.model import federated_averaging
from storage.persistence import RoundPersistence


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
class RegisterAggregator(Message):
    aggregator_id: str = "aggregator"
    host: str = "localhost"
    port: int = 0
    

class Aggregator(Actor):
    def __init__(self, provider_ref: ActorRef = None):
        super().__init__()
        self.provider_ref: Optional[ActorRef] = provider_ref
        self.persistence = RoundPersistence()

    async def pre_start(self):
        if self.provider_ref:
            advertised_host = "localhost"
            if self.context._system.host not in ("0.0.0.0", "127.0.0.1", "localhost"):
                advertised_host = self.context._system.host

            self.provider_ref.tell(RegisterAggregator(
                aggregator_id=self.actor_id,
                host=advertised_host,
                port=self.context._system.port
            ))
            self.log.info("Registered with provider")

    async def receive(self, msg: Message):
        if isinstance(msg, HealthPing):
            # Dual handling: respond via msg.sender (for local Supervisor) or provider_ref (for remote Provider)
            if msg.sender:
                # Respond to whoever sent the ping (works for local Supervisor)
                msg.sender.tell(HealthAck(actor_id=self.actor_id, status="alive"))
            elif self.provider_ref:
                # Fallback to provider_ref for remote pings
                self.provider_ref.tell(HealthAck(actor_id=self.actor_id, status="alive"))
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

            self.persistence.save_round(
                msg.round_idx,
                weights=aggregated,
                train_metrics={
                    "train_avg_loss": train_avg_loss,
                    "train_avg_accuracy": train_avg_acc
                }
            )

            self.provider_ref.tell(result)
            self.log.info(f"Aggregated round {msg.round_idx}: "
                          f"train_avg_loss={train_avg_loss:.4f}, "
                          f"train_avg_accuracy={train_avg_acc:.4f}")
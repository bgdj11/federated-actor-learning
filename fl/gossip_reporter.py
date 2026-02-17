
#Reporter is observer-only

import os
import sys
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import Actor, ActorRef
from actor.messages import Message, GossipState, HealthAck, Shutdown


@dataclass
class GossipReporterConfig:

    reporter_id: str = "reporter-1"
    evaluator_ref: Optional[ActorRef] = None
    eval_interval: int = 10
    startup_delay_sec: float = 2.0
    report_interval_sec: float = 5.0
    gossip_log_every: int = 10


class GossipReporter(Actor):
    def __init__(self, config: GossipReporterConfig):
        super().__init__()
        self.config = config

        self.peer_status: Dict[str, dict] = {}
        self.global_round = 0

        self.total_gossips: int = 0
        self.start_time = datetime.now()
        self._monitor_task: Optional[asyncio.Task] = None

    async def pre_start(self):
        self.log.info(f"GossipReporter STARTED: {self.config.reporter_id}")
        self.start_time = datetime.now()
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self.log.info("Reporter is monitoring... (not controlling!)")

    async def receive(self, msg: Message):
        if isinstance(msg, GossipState):
            await self._handle_peer_update(msg)
        elif isinstance(msg, HealthAck):
            await self._handle_health_ack(msg)
        elif isinstance(msg, Shutdown):
            self.log.info("Reporter shutting down...")
            await self._shutdown()

    async def _handle_peer_update(self, msg: GossipState):
        self.total_gossips += 1

        if msg.peer_id not in self.peer_status:
            self.peer_status[msg.peer_id] = {}

        self.peer_status[msg.peer_id].update(
            {
                "round": msg.round_num,
                "last_seen": asyncio.get_event_loop().time(),
                "delta_count": len(msg.crdt_deltas),
                "delta_norm": msg.delta_norm,
            }
        )

        if msg.round_num > self.global_round:
            self.global_round = msg.round_num

        self.log.debug(
            f"Peer {msg.peer_id} in round {msg.round_num} "
            f"(deltas={len(msg.crdt_deltas)}, delta_norm={msg.delta_norm:.6f})"
        )

        if self.config.gossip_log_every > 0 and self.total_gossips % self.config.gossip_log_every == 0:
            self.log.info(
                f"Received {self.total_gossips} gossip updates so far "
                f"(latest: peer={msg.peer_id}, round={msg.round_num}, delta_norm={msg.delta_norm:.6f})"
            )

    async def _handle_health_ack(self, msg: HealthAck):
        if msg.actor_id not in self.peer_status:
            self.peer_status[msg.actor_id] = {}

        self.peer_status[msg.actor_id]["last_seen"] = asyncio.get_event_loop().time()
        self.peer_status[msg.actor_id]["status"] = msg.status

    async def _monitoring_loop(self):
        await asyncio.sleep(self.config.startup_delay_sec)

        while True:
            try:
                await asyncio.sleep(self.config.report_interval_sec)
                elapsed = (datetime.now() - self.start_time).total_seconds()

                self.log.info(f"\n=== Gossip Network Report (elapsed: {elapsed:.1f}s) ===")
                self.log.info(f"Global Round: {self.global_round}")
                self.log.info(f"Total Gossips: {self.total_gossips}")
                self.log.info(f"Active Peers: {len(self.peer_status)}")

                for peer_id, status in self.peer_status.items():
                    self.log.info(
                        f"  {peer_id}: round={status.get('round', '?')} "
                        f"status={status.get('status', '?')} "
                        f"delta_norm={status.get('delta_norm', 0.0):.6f}"
                    )

                if self.config.evaluator_ref and self.global_round % self.config.eval_interval == 0:
                    self.log.info(f"Triggering evaluation for global round {self.global_round}")

                self.log.info("=== End Report ===\n")

            except Exception as e:
                self.log.error(f"Monitoring error: {e}")

    async def _shutdown(self):
        if self._monitor_task:
            self._monitor_task.cancel()

        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log.info("Reporter stopped. Summary:")
        self.log.info(f"  Duration: {elapsed:.1f}s")
        self.log.info(f"  Global Round: {self.global_round}")
        self.log.info(f"  Total Gossips: {self.total_gossips}")
        self.log.info(f"  Final Peers: {len(self.peer_status)}")

    async def post_stop(self):
        await self._shutdown()

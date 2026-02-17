"""
GossipPeer - autonomous P2P actor with CRDT gossip protocol.

Key behavior:
- Gossip exchanges CRDT snapshots (state-based merge).
- Each peer publishes model under model/<peer_id>.
- Global model is recomputed via async weighted FedAvg.
- Peer can restore CRDT snapshot from persistence at startup.
"""

import os
import sys
import asyncio
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import Actor, ActorRef
from actor.messages import (
    Message,
    GossipPeerJoin,
    GossipState,
    MembershipUpdate,
    HealthPing,
    HealthAck,
    Shutdown,
)
from fl.crdt import LWWMap, PNCounter
from fl.model import SimpleClassifier, federated_averaging
from storage.persistence import GossipPersistence


@dataclass
class GossipConfig:
    peer_id: str
    fanout: int = 2
    gossip_interval: float = 3.0
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01

    convergence_eps: float = 0.001
    convergence_patience: int = 3
    max_rounds: Optional[int] = None

    reporter_host: Optional[str] = None
    reporter_port: Optional[int] = None
    seed_peers: List[Tuple[str, int]] = field(default_factory=list)

    min_global_apply_eps: float = 1e-9


class GossipPeer(Actor):

    MODEL_KEY_PREFIX = "model/"

    def __init__(self, config: GossipConfig, data: Tuple[np.ndarray, np.ndarray] = None):
        super().__init__()
        self.config = config
        self.X, self.y = data if data else (None, None)

        self.lww_map = LWWMap(config.peer_id)
        self.pn_counter = PNCounter(config.peer_id)

        input_dim = int(self.X.shape[1]) if self.X is not None else 512
        num_classes = int(np.max(self.y) + 1) if self.y is not None and len(self.y) > 0 else 10
        self.model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes, lr=config.lr)

        self.round_num = 0
        self.start_round = 0
        self.n_samples = int(len(self.y)) if self.y is not None else 0

        # Membership / routing
        self.known_peers: Dict[str, ActorRef] = {}
        self.peer_info: Dict[str, dict] = {}  # peer_id -> {host, port, last_seen}
        self.seed_refs: Dict[str, ActorRef] = {}  # endpoint host:port -> ActorRef (temporary)
        self.reporter_ref: Optional[ActorRef] = None

        self.last_delta_norm = 0.0
        self.convergence_count = 0
        self.stopped = False

        self._self_host = "localhost"
        self._self_port = 0

        self._last_global_weights: Optional[Dict[str, np.ndarray]] = None

        self._persistence = GossipPersistence()

        self._gossip_task: Optional[asyncio.Task] = None
        self._training_task: Optional[asyncio.Task] = None

    async def pre_start(self):
        self.log.info(f"Gossip Peer STARTED (AUTONOMOUS): {self.config.peer_id}")

        try:
            snap = self._persistence.load_latest_crdt_snapshot(self.config.peer_id)
            if snap:
                self.round_num = int(snap["round_num"])
                self.lww_map = LWWMap.from_dict(snap["lww_state"])
                self.pn_counter = PNCounter.from_dict(snap["pn_state"])

                self.lww_map.replica_id = self.config.peer_id
                self.pn_counter.replica_id = self.config.peer_id

                self.log.info(
                    f"Restored CRDT snapshot for {self.config.peer_id}: round={self.round_num}"
                )
        except Exception as e:
            self.log.warning(f"Could not restore persistence snapshot: {e}")

        self.start_round = int(self.round_num)

        # 1) Resolve local endpoint and seed endpoints
        system_host = self.context._system.host
        if system_host in ("0.0.0.0", "127.0.0.1", "localhost"):
            self._self_host = "localhost"
        else:
            self._self_host = system_host
        self._self_port = int(self.context._system.port)

        now = asyncio.get_event_loop().time()
        self.peer_info[self.config.peer_id] = {
            "host": self._self_host,
            "port": self._self_port,
            "last_seen": now,
        }

        for host, port in self.config.seed_peers:
            endpoint = f"{host}:{int(port)}"
            if host == self._self_host and int(port) == self._self_port:
                continue
            if endpoint not in self.seed_refs:
                self.seed_refs[endpoint] = self.context._system.remote_ref("peer", host, int(port))

        # 2) Base CRDT state
        self.lww_map.put("peer_id", self.config.peer_id)
        self.lww_map.put("status", "active")
        self.lww_map.put(
            "model_meta",
            {"input_dim": self.model.input_dim, "num_classes": self.model.num_classes},
        )
        self.pn_counter.increment()

        # 3) Publish current model immediately
        self._publish_local_model(round_num=self.round_num)

        # 4) Apply global model if available in restored state
        self._recompute_and_apply_global_model(reason="startup")

        # 5) Start background loops
        self._training_task = asyncio.create_task(self._autonomous_training_loop())
        self._gossip_task = asyncio.create_task(self._autonomous_gossip_loop())

        # 6) Optional reporter
        if self.config.reporter_host and self.config.reporter_port:
            self.reporter_ref = self.context._system.remote_ref(
                "reporter", self.config.reporter_host, self.config.reporter_port
            )
            self.log.info(
                f"Reporter connected at {self.config.reporter_host}:{self.config.reporter_port}"
            )

        self.log.info(f"Peer {self.config.peer_id} is RUNNING autonomously")

    async def receive(self, msg: Message):
        if isinstance(msg, GossipPeerJoin):
            await self._register_peer(msg)
        elif isinstance(msg, MembershipUpdate):
            return
        elif isinstance(msg, GossipState):
            await self._handle_gossip_state(msg)
        elif isinstance(msg, HealthPing):
            if msg.sender:
                msg.sender.tell(HealthAck(actor_id=self.actor_id, status="active"))
        elif isinstance(msg, Shutdown):
            self.log.info(f"Peer {self.config.peer_id} shutting down...")
            await self._shutdown()

    async def post_stop(self):
        await self._shutdown()

    async def _shutdown(self):
        self.stopped = True
        if self._gossip_task:
            self._gossip_task.cancel()
        if self._training_task:
            self._training_task.cancel()

        self.log.info(
            f"Peer {self.config.peer_id} stopped after {self.round_num} rounds. "
            f"Final delta_norm={self.last_delta_norm:.6f}"
        )

    async def _register_peer(self, msg: GossipPeerJoin):
        if not msg.peer_id or ":" in msg.peer_id or msg.peer_id == self.config.peer_id:
            return

        host = str(msg.host)
        port = int(msg.port)
        if host == self._self_host and port == self._self_port:
            return

        self.log.info(f"Discovered peer: {msg.peer_id} at {host}:{port}")
        self.peer_info[msg.peer_id] = {
            "host": host,
            "port": port,
            "last_seen": asyncio.get_event_loop().time(),
        }
        self.known_peers[msg.peer_id] = self.context._system.remote_ref("peer", host, port)

        self.seed_refs.pop(f"{host}:{port}", None)

    async def _handle_gossip_state(self, msg: GossipState):
        self.log.debug(f"Merge gossip from {msg.peer_id}: {len(msg.crdt_deltas)} deltas")

        for delta in msg.crdt_deltas:
            if delta.get("type") == "lww":
                self.lww_map.merge_state(delta.get("data", {}))
            elif delta.get("type") == "pn":
                self.pn_counter.merge(delta.get("data", {}))

        # Membership piggybacking
        for peer_id, info in (msg.peer_info or {}).items():
            if not isinstance(info, dict):
                continue
            host = info.get("host")
            port = info.get("port")
            if not peer_id or ":" in peer_id or not host or port is None:
                continue

            port = int(port)
            if peer_id == self.config.peer_id:
                continue
            if host == self._self_host and port == self._self_port:
                continue

            if peer_id not in self.known_peers:
                self.log.info(f"Learned about new peer {peer_id} at {host}:{port} via gossip")
                self.peer_info[peer_id] = {
                    "host": host,
                    "port": port,
                    "last_seen": asyncio.get_event_loop().time(),
                }
                self.known_peers[peer_id] = self.context._system.remote_ref("peer", host, port)
                self.seed_refs.pop(f"{host}:{port}", None)

        if msg.peer_id in self.peer_info:
            self.peer_info[msg.peer_id]["last_seen"] = asyncio.get_event_loop().time()

        self._recompute_and_apply_global_model(reason=f"gossip_from={msg.peer_id}")

    async def _autonomous_gossip_loop(self):
        await asyncio.sleep(2.0)

        while not self.stopped:
            try:
                await asyncio.sleep(self.config.gossip_interval)

                deltas = [
                    {"type": "lww", "data": self.lww_map.to_dict()},
                    {"type": "pn", "data": self.pn_counter.to_dict()},
                ]

                peer_info_to_send = {
                    peer_id: {"host": info["host"], "port": int(info["port"])}
                    for peer_id, info in self.peer_info.items()
                    if peer_id and ":" not in peer_id
                }

                delta_norm = float(self.last_delta_norm)
                self._update_convergence(delta_norm)

                target_refs: Dict[str, ActorRef] = {}
                for peer_id, peer_ref in self.known_peers.items():
                    if peer_id and peer_id != self.config.peer_id:
                        target_refs[peer_id] = peer_ref

                for endpoint, peer_ref in self.seed_refs.items():
                    host, _, port_str = endpoint.partition(":")
                    try:
                        port = int(port_str)
                    except ValueError:
                        continue
                    if host == self._self_host and port == self._self_port:
                        continue
                    target_refs[endpoint] = peer_ref

                target_count = min(self.config.fanout, len(target_refs))
                sent_to: List[str] = []
                if target_count > 0:
                    targets = random.sample(list(target_refs.keys()), target_count)
                    for target_key in targets:
                        peer_ref = target_refs.get(target_key)
                        if peer_ref:
                            peer_ref.tell(
                                GossipState(
                                    peer_id=self.config.peer_id,
                                    round_num=self.round_num,
                                    delta_norm=delta_norm,
                                    crdt_deltas=deltas,
                                    peer_info=peer_info_to_send,
                                )
                            )
                            sent_to.append(target_key)

                if sent_to:
                    self.log.info(
                        f"Gossip sent to [{', '.join(sent_to)}] delta_norm={delta_norm:.6f}"
                    )

                if self.reporter_ref:
                    self.reporter_ref.tell(
                        GossipState(
                            peer_id=self.config.peer_id,
                            round_num=self.round_num,
                            delta_norm=delta_norm,
                            crdt_deltas=deltas,
                            peer_info=peer_info_to_send,
                        )
                    )

                if self._is_converged():
                    self.log.info(
                        f"CONVERGENCE DETECTED after {self.config.convergence_patience} flushes"
                    )
                    self.stopped = True
                    break

            except Exception as e:
                self.log.error(f"Gossip error: {e}")

    async def _autonomous_training_loop(self):
        await asyncio.sleep(1.0)

        if self.X is None or self.y is None:
            self.log.warning("No training data - skipping training loop")
            return

        while not self.stopped and self._can_run_more_rounds():
            try:
                self._recompute_and_apply_global_model(reason="before_train")

                self.round_num += 1

                prev_global = self._last_global_weights
                prev_local = self.model.get_weights()

                total_loss = 0.0
                total_acc = 0.0
                for _ in range(self.config.local_epochs):
                    metrics = self.model.train_epoch(self.X, self.y, self.config.batch_size)
                    total_loss += metrics.get("loss", 0.0)
                    total_acc += metrics.get("accuracy", 0.0)

                avg_metrics = {
                    "loss": total_loss / max(self.config.local_epochs, 1),
                    "accuracy": total_acc / max(self.config.local_epochs, 1),
                }

                self._publish_local_model(round_num=self.round_num)
                self._recompute_and_apply_global_model(reason="after_local_publish")

                if prev_global is not None and self._last_global_weights is not None:
                    self.last_delta_norm = self._compute_weight_delta_norm(
                        prev_global, self._last_global_weights
                    )
                else:
                    self.last_delta_norm = self._compute_weight_delta_norm(
                        prev_local, self.model.get_weights()
                    )

                self.log.info(
                    f"Round {self.round_num} global_delta_norm={self.last_delta_norm:.6f} "
                    f"loss={avg_metrics['loss']:.4f} acc={avg_metrics['accuracy']:.4f}"
                )

                try:
                    self._persistence.save_crdt_snapshot(
                        self.config.peer_id,
                        self.round_num,
                        self.lww_map.to_dict(),
                        self.pn_counter.to_dict(),
                    )
                    self._persistence.save_peer_metrics(
                        self.config.peer_id,
                        self.round_num,
                        avg_metrics,
                    )
                except Exception as e:
                    self.log.warning(f"Persistence save failed: {e}")

                await asyncio.sleep(2.0)

            except Exception as e:
                self.log.error(f"Training error: {e}")
                await asyncio.sleep(1.0)

    def _encode_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        W = weights["W"]
        b = weights["b"]
        return {
            "W_hex": W.tobytes().hex(),
            "b_hex": b.tobytes().hex(),
            "W_shape": list(W.shape),
            "b_shape": list(b.shape),
            "dtype": str(W.dtype),
        }

    def _decode_weights(self, payload: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        try:
            dtype = np.dtype(payload.get("dtype", "float64"))
            W_shape = tuple(payload["W_shape"])
            b_shape = tuple(payload["b_shape"])
            W = np.frombuffer(bytes.fromhex(payload["W_hex"]), dtype=dtype).reshape(W_shape)
            b = np.frombuffer(bytes.fromhex(payload["b_hex"]), dtype=dtype).reshape(b_shape)
            return {"W": W, "b": b}
        except Exception:
            return None

    def _publish_local_model(self, round_num: int):
        weights = self.model.get_weights()
        value = {
            "peer_id": self.config.peer_id,
            "round": int(round_num),
            "n_samples": int(self.n_samples),
            **self._encode_weights(weights),
        }
        self.lww_map.put(f"{self.MODEL_KEY_PREFIX}{self.config.peer_id}", value)
        self.pn_counter.increment()

    def _collect_peer_models(self) -> List[Tuple[Dict[str, np.ndarray], int]]:
        updates: List[Tuple[Dict[str, np.ndarray], int]] = []
        for key, (val, _) in self.lww_map.data.items():
            if not isinstance(key, str) or not key.startswith(self.MODEL_KEY_PREFIX):
                continue
            if not isinstance(val, dict):
                continue
            decoded = self._decode_weights(val)
            if decoded is None:
                continue

            if decoded["W"].shape != self.model.W.shape or decoded["b"].shape != self.model.b.shape:
                continue

            n = int(val.get("n_samples", 0))
            if n <= 0:
                n = 1
            updates.append((decoded, n))
        return updates

    def _recompute_and_apply_global_model(self, reason: str = ""):
        updates = self._collect_peer_models()
        if not updates:
            return

        try:
            global_weights = federated_averaging(updates)
        except Exception:
            return

        delta_norm = 0.0
        if self._last_global_weights is not None:
            delta_norm = self._compute_weight_delta_norm(self._last_global_weights, global_weights)

        if self._last_global_weights is None or delta_norm > self.config.min_global_apply_eps:
            self.model.set_weights(global_weights)
            self._last_global_weights = {
                "W": global_weights["W"].copy(),
                "b": global_weights["b"].copy(),
            }
            self.last_delta_norm = float(delta_norm)
            self.log.debug(f"Applied global model (reason={reason}) delta_norm={delta_norm:.6f}")

    def _compute_weight_delta_norm(
        self,
        prev_weights: Dict[str, np.ndarray],
        new_weights: Dict[str, np.ndarray],
    ) -> float:
        diff_w = new_weights["W"] - prev_weights["W"]
        diff_b = new_weights["b"] - prev_weights["b"]
        return float(np.sqrt(np.sum(diff_w * diff_w) + np.sum(diff_b * diff_b)))

    def _can_run_more_rounds(self) -> bool:
        if self.config.max_rounds is None:
            return True
        completed_since_start = int(self.round_num) - int(self.start_round)
        return completed_since_start < int(self.config.max_rounds)

    def _update_convergence(self, delta_norm: float):
        if not np.isfinite(delta_norm):
            delta_norm = 0.0
        if delta_norm < self.config.convergence_eps:
            self.convergence_count += 1
            self.log.info(
                f"Convergence counter: {self.convergence_count}/{self.config.convergence_patience}"
            )
        else:
            self.convergence_count = 0
        self.last_delta_norm = float(delta_norm)

    def _is_converged(self) -> bool:
        return self.convergence_count >= self.config.convergence_patience

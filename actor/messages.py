from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import numpy as np
import uuid

if TYPE_CHECKING:
    from actor.actor_system import ActorRef, ActorSystem


@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    _sender_id: Optional[str] = None
    _system: Optional['ActorSystem'] = field(default=None, repr=False, compare=False)
    
    @property
    def sender(self) -> Optional['ActorRef']:
        if self._sender_id and self._system:
            from actor.actor_system import ActorRef
            return ActorRef(self._sender_id, self._system)
        return None


@dataclass
class TrainRequest(Message):
    round_idx: int = 0
    global_weights: Optional[np.ndarray] = None
    mu: float = 0.0


@dataclass
class ModelUpdate(Message):
    worker_id: str = ""
    weights: Optional[np.ndarray] = None
    num_samples: int = 0
    metrics: dict = field(default_factory=dict)


@dataclass
class GlobalModelBroadcast(Message):
    round_idx: int = 0
    weights: Optional[np.ndarray] = None


@dataclass
class HealthPing(Message):
    pass


@dataclass
class HealthAck(Message):
    actor_id: str = ""
    status: str = "alive"


@dataclass
class ChildFailed(Message):
    child_id: str = ""
    error: str = ""


@dataclass
class RestartChild(Message):
    child_id: str = ""


@dataclass
class Shutdown(Message):
    pass

@dataclass
class GossipPeerJoin(Message):
    peer_id: str = ""
    host: str = "localhost"
    port: int = 0


@dataclass
class GossipState(Message):
    peer_id: str = ""
    round_num: int = 0
    delta_norm: float = 0.0
    crdt_deltas: list = field(default_factory=list) 
    peer_info: dict = field(default_factory=dict) 


@dataclass
class MembershipUpdate(Message):
    active_peers: list = field(default_factory=list)
    timestamp: float = 0.0

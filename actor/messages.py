from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import uuid


@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class TrainRequest(Message):
    round_idx: int = 0
    global_weights: Optional[np.ndarray] = None


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
class PeerJoin(Message):
    peer_id: str = ""
    host: str = "localhost"
    port: int = 0


@dataclass
class PeerLeave(Message):
    peer_id: str = ""


@dataclass 
class PeerReady(Message):
    peer_id: str = ""
    round_idx: int = 0


@dataclass
class StartRound(Message):
    round_idx: int = 0


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
class CRDTDelta(Message):
    crdt_type: str = ""
    key: str = ""
    delta: Any = None


@dataclass
class Shutdown(Message):
    pass

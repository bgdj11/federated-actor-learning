from dataclasses import dataclass
from typing import Any, Dict, Optional, Generic, TypeVar

T = TypeVar('T')


@dataclass(frozen=True)
class Timestamp:
    replica_id: str
    logical_time: int

    def _key(self):
        return (self.logical_time, self.replica_id)

    def __lt__(self, other: 'Timestamp') -> bool:
        return self._key() < other._key()

    def __le__(self, other: 'Timestamp') -> bool:
        return self._key() <= other._key()

    def __gt__(self, other: 'Timestamp') -> bool:
        return self._key() > other._key()

    def __ge__(self, other: 'Timestamp') -> bool:
        return self._key() >= other._key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timestamp):
            return False
        return self._key() == other._key()

    def to_dict(self) -> dict:
        return {
            'replica_id': self.replica_id,
            'logical_time': int(self.logical_time),
        }

    @staticmethod
    def from_dict(d: dict) -> 'Timestamp':
        return Timestamp(
            replica_id=d['replica_id'],
            logical_time=int(d['logical_time']),
        )


class LWWMap(Generic[T]):
    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.clock = 0
        self.data: Dict[str, tuple[T, Timestamp]] = {}
        self.tombstones: Dict[str, Timestamp] = {}

    def _next_ts(self) -> Timestamp:
        self.clock += 1
        return Timestamp(self.replica_id, self.clock)

    def put(self, key: str, value: T) -> Dict[str, Any]:
        ts = self._next_ts()
        self.data[key] = (value, ts)

        if key in self.tombstones and self.tombstones[key] < ts:
            del self.tombstones[key]

        return {'type': 'put', 'key': key, 'value': value, 'ts': ts.to_dict()}

    def get(self, key: str) -> Optional[T]:
        if key not in self.data:
            return None

        value, ts = self.data[key]
        if key in self.tombstones and self.tombstones[key] >= ts:
            return None

        return value

    def delete(self, key: str) -> Dict[str, Any]:
        ts = self._next_ts()
        self.tombstones[key] = ts
        return {'type': 'delete', 'key': key, 'ts': ts.to_dict()}

    def merge(self, other_delta: dict):
        delta_type = other_delta.get('type')

        if delta_type == 'put':
            key = other_delta['key']
            value = other_delta['value']
            ts = Timestamp.from_dict(other_delta['ts'])

            if key not in self.data or ts > self.data[key][1]:
                self.data[key] = (value, ts)
                if key in self.tombstones and ts > self.tombstones[key]:
                    del self.tombstones[key]

            self.clock = max(self.clock, ts.logical_time) + 1

        elif delta_type == 'delete':
            key = other_delta['key']
            ts = Timestamp.from_dict(other_delta['ts'])

            if key not in self.tombstones or ts > self.tombstones[key]:
                self.tombstones[key] = ts
                if key in self.data and ts >= self.data[key][1]:
                    del self.data[key]

            self.clock = max(self.clock, ts.logical_time) + 1

    def merge_state(self, other_state: dict):
        if 'type' in other_state and other_state.get('type') in ('put', 'delete'):
            self.merge(other_state)
            return

        other_clock = int(other_state.get('clock', 0))
        self.clock = max(self.clock, other_clock) + 1

        other_tomb = other_state.get('tombstones', {}) or {}
        for key, ts_dict in other_tomb.items():
            ots = Timestamp.from_dict(ts_dict)
            if key not in self.tombstones or ots > self.tombstones[key]:
                self.tombstones[key] = ots
            if key in self.data and self.tombstones[key] >= self.data[key][1]:
                del self.data[key]

        other_data = other_state.get('data', {}) or {}
        for key, entry in other_data.items():
            try:
                ots = Timestamp.from_dict(entry['ts'])
            except Exception:
                continue
            value = entry.get('value')

            if key in self.tombstones and self.tombstones[key] >= ots:
                continue

            if key not in self.data or ots > self.data[key][1]:
                self.data[key] = (value, ots)

    def get_tombstones(self) -> Dict[str, Timestamp]:
        old_tombstones: Dict[str, Timestamp] = {}

        for key, ts in list(self.tombstones.items()):
            if self.clock - ts.logical_time > 300:
                old_tombstones[key] = ts
                if key in self.data:
                    data_ts = self.data[key][1]
                    if ts >= data_ts:
                        del self.data[key]
                del self.tombstones[key]

        return old_tombstones

    def to_dict(self) -> dict:
        data_dict: Dict[str, Any] = {}
        for k, (v, ts) in self.data.items():
            data_dict[k] = {'value': v, 'ts': ts.to_dict()}

        tombstone_dict: Dict[str, Any] = {}
        for k, ts in self.tombstones.items():
            tombstone_dict[k] = ts.to_dict()

        return {
            'replica_id': self.replica_id,
            'clock': int(self.clock),
            'data': data_dict,
            'tombstones': tombstone_dict,
        }

    @staticmethod
    def from_dict(d: dict) -> 'LWWMap':
        lww = LWWMap(d['replica_id'])
        lww.clock = int(d.get('clock', 0))

        for k, v in (d.get('data', {}) or {}).items():
            ts = Timestamp.from_dict(v['ts'])
            lww.data[k] = (v.get('value'), ts)

        for k, ts_data in (d.get('tombstones', {}) or {}).items():
            lww.tombstones[k] = Timestamp.from_dict(ts_data)

        return lww


class PNCounter:

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.P: Dict[str, int] = {replica_id: 0}
        self.N: Dict[str, int] = {replica_id: 0}

    def increment(self) -> Dict[str, Any]:
        self.P[self.replica_id] += 1
        return {
            'type': 'increment',
            'replica_id': self.replica_id,
            'P': dict(self.P),
            'N': dict(self.N),
        }

    def decrement(self) -> Dict[str, Any]:
        self.N[self.replica_id] += 1
        return {
            'type': 'decrement',
            'replica_id': self.replica_id,
            'P': dict(self.P),
            'N': dict(self.N),
        }

    def value(self) -> int:
        return sum(self.P.values()) - sum(self.N.values())

    def merge(self, other_delta: dict):
        other_p = other_delta.get('P', {}) or {}
        other_n = other_delta.get('N', {}) or {}

        for rid, count in other_p.items():
            self.P[rid] = max(self.P.get(rid, 0), int(count))

        for rid, count in other_n.items():
            self.N[rid] = max(self.N.get(rid, 0), int(count))

    def to_dict(self) -> dict:
        return {
            'replica_id': self.replica_id,
            'P': dict(self.P),
            'N': dict(self.N),
            'value': int(self.value()),
        }

    @staticmethod
    def from_dict(d: dict) -> 'PNCounter':
        counter = PNCounter(d['replica_id'])
        counter.P = dict(d.get('P', {}))
        counter.N = dict(d.get('N', {}))
        return counter

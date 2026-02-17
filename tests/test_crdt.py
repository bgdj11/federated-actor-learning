import pytest
import time
from fl.crdt import LWWMap, PNCounter, Timestamp


class TestTimestamp:
    def test_timestamp_comparison_uses_logical_time(self):
        ts1 = Timestamp("node1", 1)
        ts2 = Timestamp("node1", 2)

        assert ts1 < ts2
        assert ts2 > ts1

    def test_timestamp_tie_breaker_replica_id(self):
        ts1 = Timestamp("A", 10)
        ts2 = Timestamp("B", 10)

        assert ts1 != ts2
        assert (ts1 < ts2) == ("A" < "B")

    def test_timestamp_serialization_roundtrip(self):
        ts = Timestamp("node1", 42)
        d = ts.to_dict()
        ts2 = Timestamp.from_dict(d)

        assert ts == ts2


class TestLWWMap:
    def test_put_get_delete(self):
        lww = LWWMap("node1")
        lww.put("k", "v")
        assert lww.get("k") == "v"

        lww.delete("k")
        assert lww.get("k") is None

    def test_merge_delta_put(self):
        a = LWWMap("A")
        b = LWWMap("B")

        d = a.put("x", 1)
        b.merge(d)
        assert b.get("x") == 1

    def test_merge_state_snapshot(self):
        a = LWWMap("A")
        b = LWWMap("B")

        a.put("k1", {"a": 1})
        snap = a.to_dict()
        b.merge_state(snap)

        assert b.get("k1") == {"a": 1}

    def test_delete_wins_over_older_put(self):
        a = LWWMap("A")
        b = LWWMap("B")

        put_delta = a.put("k", "v1")
        time.sleep(0.001)
        del_delta = b.delete("k")

        b.merge(put_delta)
        b.merge(del_delta)
        assert b.get("k") is None

        c = LWWMap("C")
        c.merge(del_delta)
        c.merge(put_delta)
        assert c.get("k") is None


class TestPNCounter:
    def test_increment_decrement_and_merge(self):
        c1 = PNCounter("n1")
        c2 = PNCounter("n2")

        c1.increment()
        c1.increment()
        c2.increment()
        c2.decrement()

        snap1 = c1.to_dict()
        snap2 = c2.to_dict()

        c1.merge(snap2)
        c2.merge(snap1)

        assert c1.value() == c2.value()

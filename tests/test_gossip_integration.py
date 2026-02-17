import pytest
import asyncio
import numpy as np

from actor.actor_system import ActorSystem
from actor.messages import GossipPeerJoin
from fl.gossip_peer import GossipPeer, GossipConfig


@pytest.mark.asyncio
async def test_two_systems_join_and_gossip_exchange_models():
    sys_a = ActorSystem("sys-A", host="localhost", port=0)
    sys_b = ActorSystem("sys-B", host="localhost", port=0)

    await sys_a.start_server()
    await sys_b.start_server()

    X = np.zeros((3, 8), dtype=np.float64)
    y = np.array([0, 1, 2], dtype=np.int64)

    cfg_a = GossipConfig(peer_id="peer-A", gossip_interval=0.2, fanout=1, max_rounds=0, seed_peers=[("localhost", sys_b.port)])
    cfg_b = GossipConfig(peer_id="peer-B", gossip_interval=0.2, fanout=1, max_rounds=0, seed_peers=[("localhost", sys_a.port)])

    ref_a = sys_a.actor_of(GossipPeer, "peer", config=cfg_a, data=(X, y))
    ref_b = sys_b.actor_of(GossipPeer, "peer", config=cfg_b, data=(X, y))

    remote_b_from_a = sys_a.remote_ref("peer", "localhost", sys_b.port)
    remote_a_from_b = sys_b.remote_ref("peer", "localhost", sys_a.port)

    remote_b_from_a.tell(GossipPeerJoin(peer_id="peer-A", host="localhost", port=sys_a.port), sender=ref_a)
    remote_a_from_b.tell(GossipPeerJoin(peer_id="peer-B", host="localhost", port=sys_b.port), sender=ref_b)

    await asyncio.sleep(2.6)

    peer_a = sys_a._actors["peer"]
    peer_b = sys_b._actors["peer"]

    assert peer_a.lww_map.get("model/peer-B") is not None
    assert peer_b.lww_map.get("model/peer-A") is not None

    await sys_a.shutdown()
    await sys_b.shutdown()

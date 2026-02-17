import pytest
import asyncio
import numpy as np
from actor.actor_system import ActorSystem
from actor.messages import GossipState, GossipPeerJoin
from fl.gossip_peer import GossipPeer, GossipConfig
from fl.gossip_reporter import GossipReporter, GossipReporterConfig


@pytest.mark.asyncio
async def test_autonomous_peer_creation():
    system = ActorSystem("test-auto-create", host="localhost", port=0)
    
    X = np.random.randn(100, 64)
    y = np.random.randint(0, 10, 100)
    
    config = GossipConfig(
        peer_id="solo-peer",
        max_rounds=5,
        gossip_interval=1.0
    )
    peer_ref = system.actor_of(GossipPeer, "peer", config=config, data=(X, y))
    peer = system._actors["peer"]
    
    assert peer.config.peer_id == "solo-peer"
    assert peer.X is not None
    assert peer.round_num == 0
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_autonomous_training_loop():
    system = ActorSystem("test-train-loop", host="localhost", port=0)
    
    X = np.random.randn(100, 64)
    y = np.random.randint(0, 10, 100)
    
    config = GossipConfig(
        peer_id="trainer-peer",
        max_rounds=10,
        gossip_interval=2.0
    )
    peer_ref = system.actor_of(GossipPeer, "peer", config=config, data=(X, y))
    peer = system._actors["peer"]
    
    await asyncio.sleep(2.5)
    
    assert peer.round_num >= 1
    assert hasattr(peer, 'last_delta_norm')
    assert hasattr(peer, 'convergence_count')
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_delta_norm_computation():
    system = ActorSystem("test-delta", host="localhost", port=0)
    
    X = np.random.randn(50, 64)
    y = np.random.randint(0, 10, 50)
    
    config = GossipConfig(peer_id="delta-peer", max_rounds=5)
    peer_ref = system.actor_of(GossipPeer, "peer", config=config, data=(X, y))
    peer = system._actors["peer"]
    
    await asyncio.sleep(2.5)
    first_delta = peer.last_delta_norm if hasattr(peer, 'last_delta_norm') else 0.0
    assert first_delta >= 0, "Delta_norm trebalo bi da bude >= 0"

    await asyncio.sleep(2.5)
    second_delta = peer.last_delta_norm if hasattr(peer, 'last_delta_norm') else 0.0

    assert peer.round_num >= 2
    assert np.isfinite(first_delta) and np.isfinite(second_delta)
    assert second_delta >= 0
    
    await system.shutdown()



@pytest.mark.asyncio
async def test_autonomous_gossip_exchange():
    system = ActorSystem("test-gossip-exchange", host="localhost", port=0)

    X = np.random.randn(50, 64)
    y = np.random.randint(0, 10, 50)

    config1 = GossipConfig(peer_id="peer-1", gossip_interval=999, max_rounds=0)
    config2 = GossipConfig(peer_id="peer-2", gossip_interval=999, max_rounds=0)

    system.actor_of(GossipPeer, "peer-1", config=config1, data=(X, y))
    system.actor_of(GossipPeer, "peer-2", config=config2, data=(X, y))

    await asyncio.sleep(0.2)
    peer1 = system._actors["peer-1"]
    peer2 = system._actors["peer-2"]

    msg_from_peer2 = GossipState(
        peer_id="peer-2",
        round_num=0,
        delta_norm=0.0,
        crdt_deltas=[
            {"type": "lww", "data": peer2.lww_map.to_dict()},
            {"type": "pn", "data": peer2.pn_counter.to_dict()},
        ],
        peer_info={"peer-2": {"host": "localhost", "port": 9999}},
    )
    await peer1._handle_gossip_state(msg_from_peer2)

    assert "peer-2" in peer1.known_peers
    assert peer1.lww_map.get("model/peer-2") is not None

    await system.shutdown()


@pytest.mark.asyncio  
async def test_gossip_state_merge_without_barrier():
    system = ActorSystem("test-crdt-merge", host="localhost", port=0)
    
    X = np.random.randn(30, 64)
    y = np.random.randint(0, 10, 30)
    
    config1 = GossipConfig(peer_id="merge-peer-1")
    config2 = GossipConfig(peer_id="merge-peer-2")
    
    peer1_ref = system.actor_of(GossipPeer, "peer-1", config=config1, data=(X, y))
    peer2_ref = system.actor_of(GossipPeer, "peer-2", config=config2, data=(X, y))
    
    peer1 = system._actors["peer-1"]
    peer2 = system._actors["peer-2"]
    
    peer1.lww_map.put("test_key", {"value": 123})
    
    gossip_msg = GossipState(
        peer_id="merge-peer-1",
        round_num=1,
        crdt_deltas=[
            {'type': 'lww', 'data': peer1.lww_map.to_dict()},
            {'type': 'pn', 'data': peer1.pn_counter.to_dict()}
        ]
    )
    
    await peer2._handle_gossip_state(gossip_msg)
    assert peer2.lww_map.get("test_key") == {"value": 123}
    await system.shutdown()


@pytest.mark.asyncio
async def test_convergence_detection():
    system = ActorSystem("test-convergence", host="localhost", port=0)
    
    X = np.random.randn(20, 64)
    y = np.random.randint(0, 10, 20)
    
    config = GossipConfig(
        peer_id="conv-peer",
        convergence_eps=0.05,
        convergence_patience=2,
        max_rounds=10
    )
    
    peer_ref = system.actor_of(GossipPeer, "peer", config=config, data=(X, y))
    peer = system._actors["peer"]
    
    peer._update_convergence(0.001)
    assert peer.convergence_count == 1
    
    peer._update_convergence(0.002)
    assert peer.convergence_count == 2

    assert peer._is_converged() == True
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_convergence_reset_on_large_delta():
    system = ActorSystem("test-conv-reset", host="localhost", port=0)
    
    X = np.random.randn(20, 64)
    y = np.random.randint(0, 10, 20)
    
    config = GossipConfig(
        peer_id="reset-peer",
        convergence_eps=0.05,
        convergence_patience=2
    )
    
    peer_ref = system.actor_of(GossipPeer, "peer", config=config, data=(X, y))
    peer = system._actors["peer"]
    
    peer._update_convergence(0.001)
    assert peer.convergence_count == 1
    
    peer._update_convergence(0.1)
    assert peer.convergence_count == 0
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_reporter_monitoring():
    system = ActorSystem("test-reporter", host="localhost", port=0)
    
    config = GossipReporterConfig(reporter_id="test-reporter")
    reporter_ref = system.actor_of(GossipReporter, "reporter", config=config)
    
    assert "reporter" in system._actors
    reporter = system._actors["reporter"]

    gossip_msg = GossipState(
        peer_id="test-peer-1",
        round_num=5,
        crdt_deltas=[
            {'type': 'lww', 'data': {}},
            {'type': 'pn', 'data': {}}
        ]
    )
    
    await reporter._handle_peer_update(gossip_msg)
    
    assert "test-peer-1" in reporter.peer_status
    assert reporter.peer_status["test-peer-1"]["round"] == 5
    assert reporter.total_gossips == 1
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_multiple_peers_autonomous():
    system = ActorSystem("test-multi-auto", host="localhost", port=0)
    
    X = np.random.randn(50, 64)
    y = np.random.randint(0, 10, 50)
    
    peer_refs = []
    for i in range(3):
        config = GossipConfig(
            peer_id=f"peer-{i}",
            gossip_interval=0.5,
            max_rounds=3
        )
        peer_ref = system.actor_of(GossipPeer, f"peer-{i}", config=config, data=(X, y))
        peer_refs.append(peer_ref)
    
    await asyncio.sleep(2.5)
    
    for i in range(3):
        peer = system._actors[f"peer-{i}"]
        assert peer.round_num > 0, f"peer-{i} nije trenirao"
    
    await system.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

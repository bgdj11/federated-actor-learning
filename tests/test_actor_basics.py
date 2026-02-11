import pytest
import asyncio
from actor.actor_system import ActorSystem

from tests.fixtures import (
    Ping, Pong, PingActor, PongActor, 
    StatefulActor, ParentActor, CounterActor
)


@pytest.mark.asyncio
async def test_local_messaging(actor_system):
    ping_ref = actor_system.actor_of(PingActor, "ping")
    pong_ref = actor_system.actor_of(PongActor, "pong")

    actor_system._actors["ping"].pong_ref = pong_ref
    actor_system._actors["pong"].ping_ref = ping_ref

    pong_ref.tell(Ping(count=1))
    await asyncio.sleep(0.5)
    ping_actor = actor_system._actors["ping"]
    assert ping_actor.received == 5, f"Expected 5 pongs, got {ping_actor.received}"


@pytest.mark.asyncio
async def test_actor_creation_and_lifecycle(actor_system):
    counter_ref = actor_system.actor_of(CounterActor, "counter")
    
    assert "counter" in actor_system._actors
    assert actor_system._actors["counter"].actor_id == "counter"
    
    for i in range(3):
        counter_ref.tell(Ping(count=i))
    
    await asyncio.sleep(0.2)
    
    counter = actor_system._actors["counter"]
    assert counter.count == 3


@pytest.mark.asyncio
async def test_state_machine_become_unbecome(actor_system):
    """Test actor state transitions using become/unbecome."""
    ref = actor_system.actor_of(StatefulActor, "stateful")
    actor = actor_system._actors["stateful"]

    # Initial state should be idle
    assert actor.state == "idle"
    
    # Send Ping to transition to working
    ref.tell(Ping())
    await asyncio.sleep(0.1)
    assert actor.state == "working", f"Expected working, got {actor.state}"
    
    # Send Pong to transition back to idle
    ref.tell(Pong())
    await asyncio.sleep(0.1)
    assert actor.state == "idle", f"Expected idle, got {actor.state}"


@pytest.mark.asyncio
async def test_actor_hierarchy():
    """Test parent-child actor relationships."""
    system = ActorSystem("test-hierarchy")
    parent_ref = system.actor_of(ParentActor, "parent")

    # Parent creates children on Ping messages
    parent_ref.tell(Ping())
    parent_ref.tell(Ping())
    await asyncio.sleep(0.3)

    parent = system._actors["parent"]
    assert parent.child_count == 2
    assert len(parent.context.children) == 2
    
    # Children should be registered in the system
    assert "child-0" in system._actors
    assert "child-1" in system._actors
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_actor_stop():
    """Test stopping individual actors."""
    system = ActorSystem("test-stop")
    ref = system.actor_of(CounterActor, "counter")
    
    assert "counter" in system._actors
    
    # Stop the actor
    system.stop("counter")
    await asyncio.sleep(0.2)
    
    assert "counter" not in system._actors
    assert "counter" not in system._mailboxes
    assert "counter" not in system._tasks
    
    await system.shutdown()


@pytest.mark.asyncio 
async def test_multiple_actors():
    """Test creating and using multiple actors."""
    system = ActorSystem("test-multiple")
    
    refs = []
    for i in range(5):
        ref = system.actor_of(CounterActor, f"counter-{i}")
        refs.append(ref)
    
    # Send messages to all actors
    for ref in refs:
        ref.tell(Ping())
    
    await asyncio.sleep(0.3)
    
    # All actors should have received one message
    for i in range(5):
        actor = system._actors[f"counter-{i}"]
        assert actor.count == 1
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_actor_isolation():
    """Test that actors maintain separate state."""
    system = ActorSystem("test-isolation")
    
    counter1 = system.actor_of(CounterActor, "counter1")
    counter2 = system.actor_of(CounterActor, "counter2")
    
    # Send different number of messages to each
    for _ in range(3):
        counter1.tell(Ping())
    
    for _ in range(7):
        counter2.tell(Ping())
    
    await asyncio.sleep(0.3)
    
    assert system._actors["counter1"].count == 3
    assert system._actors["counter2"].count == 7
    
    await system.shutdown()

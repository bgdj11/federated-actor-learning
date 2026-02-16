import pytest
import asyncio
from actor.actor_system import ActorSystem
from actor.supervisor import Supervisor, MonitorChild
from actor.messages import Message
from tests.fixtures import CounterActor


@pytest.mark.asyncio
async def test_supervisor_add_child():
    system = ActorSystem("test-supervisor")
    
    supervisor_ref = system.actor_of(Supervisor, "supervisor", health_check_interval=1.0)
    supervisor_ref.tell(MonitorChild(
        child_id="counter",
        actor_class=CounterActor
    ))
    
    await asyncio.sleep(0.3)
    
    assert "counter" in system._actors
    assert system._actors["counter"].actor_id == "counter"
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_supervisor_child_registered():
    system = ActorSystem("test-registered")
    
    supervisor = system.actor_of(Supervisor, "supervisor")
    supervisor_actor = system._actors["supervisor"]
    
    supervisor.tell(MonitorChild(
        child_id="counter",
        actor_class=CounterActor
    ))
    
    await asyncio.sleep(0.2)
    
    assert "counter" in supervisor_actor._monitored_children
    assert supervisor_actor._monitored_children["counter"]["ref"].actor_id == "counter"
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_supervisor_restart_on_manual_failure():
    system = ActorSystem("test-manual-restart")
    
    supervisor = system.actor_of(Supervisor, "supervisor")
    supervisor_actor = system._actors["supervisor"]
    
    supervisor.tell(MonitorChild(
        child_id="counter",
        actor_class=CounterActor
    ))
    
    await asyncio.sleep(0.2)
    
    counter_old = system._actors["counter"]
    old_id = id(counter_old)
    
    await supervisor_actor._restart_child("counter")
    await asyncio.sleep(0.3)
    
    counter_new = system._actors["counter"]
    assert counter_new is not None
    assert id(counter_new) != old_id
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_supervisor_multiple_children_storage():
    system = ActorSystem("test-multi")
    
    supervisor = system.actor_of(Supervisor, "supervisor")
    supervisor_actor = system._actors["supervisor"]
    
    supervisor.tell(MonitorChild(child_id="counter-0", actor_class=CounterActor))
    supervisor.tell(MonitorChild(child_id="counter-1", actor_class=CounterActor))
    supervisor.tell(MonitorChild(child_id="counter-2", actor_class=CounterActor))
    
    await asyncio.sleep(0.2)
    
    assert len(supervisor_actor._monitored_children) == 3
    assert all(f"counter-{i}" in system._actors for i in range(3))
    
    await system.shutdown()


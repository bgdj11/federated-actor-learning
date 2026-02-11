import pytest
import asyncio
from tests.fixtures import Ping, Pong, PingActor, PongActor, CounterActor


@pytest.mark.asyncio
async def test_remote_actor_ref(two_actor_systems):
    system1, system2 = two_actor_systems
    
    system1.actor_of(CounterActor, "remote-counter")
    
    remote_ref = system2.remote_ref("remote-counter", "localhost", system1.port)
    
    assert remote_ref.actor_id == "remote-counter"
    assert remote_ref._remote_addr == ("localhost", system1.port)


@pytest.mark.asyncio
async def test_remote_messaging(two_actor_systems):
    system1, system2 = two_actor_systems
    
    pong_ref = system1.actor_of(PongActor, "pong")
    ping_ref = system2.actor_of(PingActor, "ping")
    
    system2._actors["ping"].pong_ref = system2.remote_ref("pong", "localhost", system1.port)
    system1._actors["pong"].ping_ref = system1.remote_ref("ping", "localhost", system2.port)
    
    remote_pong = system2.remote_ref("pong", "localhost", system1.port)
    remote_pong.tell(Ping(count=1))
    
    await asyncio.sleep(1.0)
    
    ping_actor = system2._actors["ping"]
    assert ping_actor.received >= 1, f"Expected at least 1 pong, got {ping_actor.received}"


@pytest.mark.asyncio
async def test_bidirectional_remote_messaging(two_actor_systems):
    system1, system2 = two_actor_systems
    
    counter1 = system1.actor_of(CounterActor, "counter1")
    counter2 = system2.actor_of(CounterActor, "counter2")
    
    remote_counter1 = system2.remote_ref("counter1", "localhost", system1.port)
    remote_counter2 = system1.remote_ref("counter2", "localhost", system2.port)
    
    remote_counter1.tell(Ping(count=1))
    remote_counter1.tell(Ping(count=2))
    remote_counter2.tell(Ping(count=3))
    
    await asyncio.sleep(0.5)
    
    assert system1._actors["counter1"].count == 2
    assert system2._actors["counter2"].count == 1


@pytest.mark.asyncio
async def test_remote_connection_caching():
    from actor.actor_system import ActorSystem
    
    system1 = ActorSystem("sys1", port=0)
    system2 = ActorSystem("sys2", port=0)
    
    await system1.start_server()
    await system2.start_server()
    
    system1.actor_of(CounterActor, "counter")
    
    remote_ref = system2.remote_ref("counter", "localhost", system1.port)
    for i in range(5):
        remote_ref.tell(Ping(count=i))
    
    await asyncio.sleep(0.5)
    
    addr_key = ("localhost", system1.port)
    assert addr_key in system2._remote_connections
    
    await system1.shutdown()
    await system2.shutdown()


@pytest.mark.asyncio
async def test_server_startup():
    from actor.actor_system import ActorSystem
    
    system = ActorSystem("test", host="localhost", port=0)
    
    assert system._server is None
    assert system._running is False
    
    await system.start_server()
    
    assert system._server is not None
    assert system._running is True
    assert system.port > 0
    
    await system.shutdown()
    
    assert system._running is False

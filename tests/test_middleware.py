import pytest
import asyncio
from tests.fixtures import Ping, Pong, PingActor, PongActor, CounterActor

middleware_log = []

def logging_middleware(actor_id: str, msg):
    middleware_log.append(f"{actor_id}: {type(msg).__name__}")
    return msg


def filter_middleware(actor_id: str, msg):
    if isinstance(msg, Ping) and msg.count > 3:
        return None
    return msg


def transform_middleware(actor_id: str, msg):
    if isinstance(msg, Ping):
        msg.count = msg.count * 2
    return msg


@pytest.mark.asyncio
async def test_logging_middleware():
    from actor.actor_system import ActorSystem
    
    middleware_log.clear()
    
    system = ActorSystem("test-middleware")
    system.add_send_middleware(logging_middleware)
    
    counter_ref = system.actor_of(CounterActor, "counter")
    
    # Send messages
    for i in range(3):
        counter_ref.tell(Ping(count=i))
    
    await asyncio.sleep(0.3)
    
    assert len(middleware_log) > 0
    assert "counter: Ping" in middleware_log
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_filter_middleware():
    from actor.actor_system import ActorSystem
    
    system = ActorSystem("test-filter")
    system.add_send_middleware(filter_middleware)
    
    ping_ref = system.actor_of(PingActor, "ping")
    pong_ref = system.actor_of(PongActor, "pong")
    
    system._actors["ping"].pong_ref = pong_ref
    system._actors["pong"].ping_ref = ping_ref
   
    pong_ref.tell(Ping(count=1))
    await asyncio.sleep(0.6)
    
    ping_actor = system._actors["ping"]
    assert ping_actor.received == 3, f"Expected 3 pongs (filtered at >3), got {ping_actor.received}"
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_transform_middleware():
    from actor.actor_system import ActorSystem
    
    system = ActorSystem("test-transform")
    system.add_send_middleware(transform_middleware)
    
    counter_ref = system.actor_of(CounterActor, "counter")
    
    counter_ref.tell(Ping(count=5))
    await asyncio.sleep(0.2)
    
    counter = system._actors["counter"]
    assert counter.count == 1

    assert counter.messages[0].count == 10
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_multiple_middleware_chain():
    from actor.actor_system import ActorSystem
    
    middleware_log.clear()
    
    system = ActorSystem("test-chain")
    system.add_send_middleware(logging_middleware)
    system.add_send_middleware(transform_middleware)
    
    counter_ref = system.actor_of(CounterActor, "counter")
    
    counter_ref.tell(Ping(count=3))
    await asyncio.sleep(0.2)
    
    assert len(middleware_log) > 0
    counter = system._actors["counter"]
    assert counter.messages[0].count == 6
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_receive_middleware():
    from actor.actor_system import ActorSystem
    
    middleware_log.clear()
    
    system = ActorSystem("test-receive")
    system.add_receive_middleware(logging_middleware)
    
    counter_ref = system.actor_of(CounterActor, "counter")
    
    for i in range(3):
        counter_ref.tell(Ping(count=i))
    
    await asyncio.sleep(0.3)
    
    assert len(middleware_log) >= 3
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_middleware_can_block_all_messages():
    from actor.actor_system import ActorSystem
    
    def block_all(actor_id: str, msg):
        return None  # Block everything
    
    system = ActorSystem("test-block")
    system.add_send_middleware(block_all)
    
    counter_ref = system.actor_of(CounterActor, "counter")
    
    for i in range(5):
        counter_ref.tell(Ping(count=i))
    
    await asyncio.sleep(0.3)
    
    # No messages should have been delivered
    counter = system._actors["counter"]
    assert counter.count == 0
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_middleware_with_remote_messaging(two_actor_systems):
    system1, system2 = two_actor_systems
    
    middleware_log.clear()
    
    system2.add_send_middleware(logging_middleware)
    
    system1.actor_of(CounterActor, "remote-counter")
    
    remote_ref = system2.remote_ref("remote-counter", "localhost", system1.port)
    
    for i in range(3):
        remote_ref.tell(Ping(count=i))
    
    await asyncio.sleep(0.8)
    
    assert len(middleware_log) >= 3
    
    counter = system1._actors["remote-counter"]
    assert counter.count == 3

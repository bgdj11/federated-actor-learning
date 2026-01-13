import asyncio
import sys
import os
sys.path.insert(0, '.')

from dataclasses import dataclass
from actor.actor_system import ActorSystem, Actor, ActorRef
from actor.messages import Message


@dataclass
class Ping(Message):
    count: int = 0


@dataclass
class Pong(Message):
    count: int = 0


class PingActor(Actor):
    def __init__(self):
        super().__init__()
        self.pong_ref = None
        self.received = 0

    async def receive(self, msg):
        if isinstance(msg, Pong):
            self.received += 1
            self.log.info(f"Received pong #{msg.count}")
            if msg.count < 5:
                self.pong_ref.tell(Ping(count=msg.count + 1))


class PongActor(Actor):
    def __init__(self):
        super().__init__()
        self.ping_ref = None

    async def receive(self, msg):
        if isinstance(msg, Ping):
            self.log.info(f"Received ping #{msg.count}")
            self.ping_ref.tell(Pong(count=msg.count))


async def test_local_messaging():
    print("\n=== Test 1: Local Messaging ===\n")

    system = ActorSystem("test-local")

    ping_ref = system.actor_of(PingActor, "ping")
    pong_ref = system.actor_of(PongActor, "pong")

    system._actors["ping"].pong_ref = pong_ref
    system._actors["pong"].ping_ref = ping_ref

    pong_ref.tell(Ping(count=1))

    await asyncio.sleep(0.5)

    ping_actor = system._actors["ping"]
    assert ping_actor.received == 5, f"Expected 5 pongs, got {ping_actor.received}"
    print(f"Received {ping_actor.received} pongs - OK")

    await system.shutdown()


class StatefulActor(Actor):
    def __init__(self):
        super().__init__()
        self.state = "idle"

    async def receive(self, msg):
        if isinstance(msg, Ping):
            self.log.info(f"State: idle -> working")
            self.state = "working"
            self.become(self.working_state)

    async def working_state(self, msg):
        if isinstance(msg, Pong):
            self.log.info(f"State: working -> idle")
            self.state = "idle"
            self.unbecome()


async def test_state_machine():
    print("\n=== Test 2: State Machine (become/unbecome) ===\n")

    system = ActorSystem("test-state")
    ref = system.actor_of(StatefulActor, "stateful")
    actor = system._actors["stateful"]

    assert actor.state == "idle"
    ref.tell(Ping())
    await asyncio.sleep(0.1)
    assert actor.state == "working", f"Expected working, got {actor.state}"
    print("Ping -> state=working - OK")

    ref.tell(Pong())
    await asyncio.sleep(0.1)
    assert actor.state == "idle", f"Expected idle, got {actor.state}"
    print("Pong -> state=idle - OK")

    await system.shutdown()


class ParentActor(Actor):
    def __init__(self):
        super().__init__()
        self.child_count = 0

    async def pre_start(self):
        self.log.info("ParentActor started")

    async def receive(self, msg):
        if isinstance(msg, Ping):
            child_ref = self.context.actor_of(ChildActor, f"child-{self.child_count}")
            self.child_count += 1
            self.log.info(f"Created child, total: {self.child_count}")


class ChildActor(Actor):
    async def pre_start(self):
        self.log.info("ChildActor started")

    async def post_stop(self):
        self.log.info("ChildActor stopped")

    async def receive(self, msg):
        pass


async def test_hierarchy():
    print("\n=== Test 3: Actor Hierarchy ===\n")

    system = ActorSystem("test-hierarchy")
    parent_ref = system.actor_of(ParentActor, "parent")

    parent_ref.tell(Ping())
    parent_ref.tell(Ping())
    await asyncio.sleep(0.2)

    parent = system._actors["parent"]
    assert parent.child_count == 2
    assert len(parent.context.children) == 2
    print(f"Created {parent.child_count} children - OK")

    await system.shutdown()


async def test_remote_setup():
    print("\n=== Test 4: Remote Setup ===\n")

    system1 = ActorSystem("system-1", port=5001)
    system2 = ActorSystem("system-2", port=5002)

    await system1.start_server()
    await system2.start_server()

    print(f"System 1 running on port {system1.port}")
    print(f"System 2 running on port {system2.port}")

    pong_ref = system1.actor_of(PongActor, "pong")
    ping_ref = system2.actor_of(PingActor, "ping")

    system2._actors["ping"].pong_ref = system2.remote_ref("pong", "localhost", 5001)
    system1._actors["pong"].ping_ref = system1.remote_ref("ping", "localhost", 5002)

    remote_pong = system2.remote_ref("pong", "localhost", 5001)
    remote_pong.tell(Ping(count=1))

    await asyncio.sleep(1)

    ping_actor = system2._actors["ping"]
    print(f"Received {ping_actor.received} pongs via TCP")
    assert ping_actor.received >= 1, "Should receive at least 1 pong"
    print("Remote messaging - OK")

    await system1.shutdown()
    await system2.shutdown()


middleware_log = []

def logging_middleware(actor_id: str, msg):
    middleware_log.append(f"{actor_id}: {type(msg).__name__}")
    return msg

def filter_middleware(actor_id: str, msg):
    if isinstance(msg, Ping) and msg.count > 3:
        return None
    return msg


async def test_middleware():
    print("\n=== Test 5: Middleware ===\n")
    
    middleware_log.clear()
    system = ActorSystem("test-middleware")
    system.add_send_middleware(logging_middleware)
    
    ping_ref = system.actor_of(PingActor, "ping")
    pong_ref = system.actor_of(PongActor, "pong")
    
    system._actors["ping"].pong_ref = pong_ref
    system._actors["pong"].ping_ref = ping_ref
    
    pong_ref.tell(Ping(count=1))
    await asyncio.sleep(0.3)
    
    print(f"Middleware logged {len(middleware_log)} messages")
    assert len(middleware_log) > 0, "Middleware should log messages"
    print("Logging middleware - OK")
    
    await system.shutdown()
    
    middleware_log.clear()
    system2 = ActorSystem("test-filter")
    system2.add_send_middleware(filter_middleware)
    
    ping_ref2 = system2.actor_of(PingActor, "ping")
    pong_ref2 = system2.actor_of(PongActor, "pong")
    
    system2._actors["ping"].pong_ref = pong_ref2
    system2._actors["pong"].ping_ref = ping_ref2
    
    pong_ref2.tell(Ping(count=1))
    await asyncio.sleep(0.5)
    
    ping_actor = system2._actors["ping"]
    print(f"Received {ping_actor.received} pongs (filtered at count>3)")
    assert ping_actor.received == 3, f"Expected 3 pongs (filter at >3), got {ping_actor.received}"
    print("Filter middleware - OK")
    
    await system2.shutdown()


async def main():
    await test_local_messaging()
    await test_state_machine()
    await test_hierarchy()
    await test_remote_setup()
    await test_middleware()
    print("\n=== All tests passed ===\n")


if __name__ == "__main__":
    asyncio.run(main())
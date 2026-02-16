from dataclasses import dataclass
from actor.actor_system import Actor
from actor.messages import Message, HealthPing, HealthAck

@dataclass
class Ping(Message):
    count: int = 0


@dataclass
class Pong(Message):
    count: int = 0


@dataclass
class TestMessage(Message):
    data: str = ""


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


class CounterActor(Actor):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.messages = []

    async def receive(self, msg):
        if isinstance(msg, HealthPing):
            if msg.sender:
                msg.sender.tell(HealthAck(actor_id=self.actor_id, status="alive"))
        else:
            self.count += 1
            self.messages.append(msg)


class EchoActor(Actor): 
    def __init__(self):
        super().__init__()
        self.last_message = None

    async def receive(self, msg):
        self.last_message = msg
        self.log.info(f"Echo: {msg}")

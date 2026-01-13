import asyncio
import pickle
import logging
import ssl
from typing import Any, Callable, Optional, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .messages import Message, Shutdown

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')

MiddlewareFunc = Callable[[str, Message], Optional[Message]]


@dataclass
class ActorRef:
    actor_id: str
    _system: 'ActorSystem'
    _remote_addr: Optional[tuple] = None

    def tell(self, msg: Message):
        if self._remote_addr:
            asyncio.create_task(self._system._send_remote(self._remote_addr, self.actor_id, msg))
        else:
            asyncio.create_task(self._system._deliver_local(self.actor_id, msg))

    async def ask(self, msg: Message, timeout: float = 5.0) -> Any:
        future = asyncio.Future()
        self._system._pending_asks[msg.id] = future
        self.tell(msg)
        return await asyncio.wait_for(future, timeout)


class Mailbox:
    def __init__(self, capacity: int = 1000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=capacity)

    async def put(self, msg: Message):
        await self._queue.put(msg)

    async def get(self) -> Message:
        return await self._queue.get()

    def empty(self) -> bool:
        return self._queue.empty()


class Actor(ABC):
    def __init__(self):
        self.actor_id: str = ""
        self.context: Optional['ActorContext'] = None
        self._behavior: Callable = self.receive
        self._logger: Optional[logging.Logger] = None

    @property
    def log(self) -> logging.Logger:
        if not self._logger:
            self._logger = logging.getLogger(self.actor_id)
        return self._logger

    def become(self, behavior: Callable):
        self._behavior = behavior

    def unbecome(self):
        self._behavior = self.receive

    async def pre_start(self):
        pass

    async def post_stop(self):
        pass

    @abstractmethod
    async def receive(self, msg: Message):
        pass


class ActorContext:
    def __init__(self, system: 'ActorSystem', actor: Actor):
        self._system = system
        self._actor = actor
        self.children: dict[str, ActorRef] = {}
        self.parent: Optional[ActorRef] = None

    @property
    def self_ref(self) -> ActorRef:
        return ActorRef(self._actor.actor_id, self._system)

    def actor_of(self, actor_class: Type[Actor], actor_id: str, **kwargs) -> ActorRef:
        ref = self._system.actor_of(actor_class, actor_id, **kwargs)
        self.children[actor_id] = ref
        self._system._actors[actor_id].context.parent = self.self_ref
        return ref

    def stop(self, ref: ActorRef):
        self._system.stop(ref.actor_id)
        self.children.pop(ref.actor_id, None)


class ActorSystem:
    def __init__(self, name: str = "system", host: str = "localhost", port: int = 0):
        self.name = name
        self.host = host
        self.port = port
        self._actors: dict[str, Actor] = {}
        self._mailboxes: dict[str, Mailbox] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._pending_asks: dict[str, asyncio.Future] = {}
        self._server: Optional[asyncio.Server] = None
        self._remote_connections: dict[tuple, tuple] = {}
        self._running = False
        self._log = logging.getLogger(f"ActorSystem({name})")
        
        self._send_middleware: list[MiddlewareFunc] = []
        self._receive_middleware: list[MiddlewareFunc] = []
        
        self._ssl_context: Optional[ssl.SSLContext] = None
        self._ssl_client_context: Optional[ssl.SSLContext] = None

    def add_send_middleware(self, func: MiddlewareFunc):
        self._send_middleware.append(func)
        
    def add_receive_middleware(self, func: MiddlewareFunc):
        self._receive_middleware.append(func)

    def enable_ssl(self, certfile: str, keyfile: str, ca_cert: Optional[str] = None):
        self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._ssl_context.load_cert_chain(certfile, keyfile)
        
        self._ssl_client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if ca_cert:
            self._ssl_client_context.load_verify_locations(ca_cert)
        else:
            self._ssl_client_context.check_hostname = False
            self._ssl_client_context.verify_mode = ssl.CERT_NONE
        
        self._log.info("SSL enabled")

    def _apply_send_middleware(self, actor_id: str, msg: Message) -> Optional[Message]:
        for mw in self._send_middleware:
            msg = mw(actor_id, msg)
            if msg is None:
                return None
        return msg

    def _apply_receive_middleware(self, actor_id: str, msg: Message) -> Optional[Message]:
        for mw in self._receive_middleware:
            msg = mw(actor_id, msg)
            if msg is None:
                return None
        return msg

    def actor_of(self, actor_class: Type[Actor], actor_id: str, **kwargs) -> ActorRef:
        if actor_id in self._actors:
            return ActorRef(actor_id, self)

        actor = actor_class(**kwargs)
        actor.actor_id = actor_id
        actor.context = ActorContext(self, actor)

        self._actors[actor_id] = actor
        self._mailboxes[actor_id] = Mailbox()
        self._tasks[actor_id] = asyncio.create_task(self._run_actor(actor_id))

        self._log.info(f"Created actor: {actor_id}")
        return ActorRef(actor_id, self)

    def stop(self, actor_id: str):
        if actor_id not in self._actors:
            return
        asyncio.create_task(self._stop_actor(actor_id))

    async def _stop_actor(self, actor_id: str):
        actor = self._actors.get(actor_id)
        if not actor:
            return

        for child_id in list(actor.context.children.keys()):
            await self._stop_actor(child_id)

        self._tasks[actor_id].cancel()
        try:
            await self._tasks[actor_id]
        except asyncio.CancelledError:
            pass

        await actor.post_stop()
        del self._actors[actor_id]
        del self._mailboxes[actor_id]
        del self._tasks[actor_id]
        self._log.info(f"Stopped actor: {actor_id}")

    async def _run_actor(self, actor_id: str):
        actor = self._actors[actor_id]
        mailbox = self._mailboxes[actor_id]

        await actor.pre_start()

        while True:
            try:
                msg = await mailbox.get()
                if isinstance(msg, Shutdown):
                    break
                
                msg = self._apply_receive_middleware(actor_id, msg)
                if msg is None:
                    continue
                    
                await actor._behavior(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Actor {actor_id} error: {e}")
                if actor.context.parent:
                    from .messages import ChildFailed
                    actor.context.parent.tell(ChildFailed(child_id=actor_id, error=str(e)))

    async def _deliver_local(self, actor_id: str, msg: Message):
        msg = self._apply_send_middleware(actor_id, msg)
        if msg is None:
            return
        if actor_id in self._mailboxes:
            await self._mailboxes[actor_id].put(msg)

    async def start_server(self):
        self._server = await asyncio.start_server(
            self._handle_connection, 
            self.host, 
            self.port,
            ssl=self._ssl_context
        )
        addr = self._server.sockets[0].getsockname()
        self.port = addr[1]
        self._running = True
        ssl_status = "with SSL" if self._ssl_context else "no SSL"
        self._log.info(f"TCP server started on {self.host}:{self.port} ({ssl_status})")

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        self._log.info(f"Connection from {addr}")

        try:
            while True:
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, 'big')
                data = await reader.readexactly(length)
                actor_id, msg = pickle.loads(data)
                await self._deliver_local(actor_id, msg)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _send_remote(self, addr: tuple, actor_id: str, msg: Message):
        msg = self._apply_send_middleware(actor_id, msg)
        if msg is None:
            return
            
        try:
            key = addr
            if key not in self._remote_connections:
                reader, writer = await asyncio.open_connection(
                    addr[0], 
                    addr[1],
                    ssl=self._ssl_client_context
                )
                self._remote_connections[key] = (reader, writer)
            else:
                _, writer = self._remote_connections[key]

            data = pickle.dumps((actor_id, msg))
            writer.write(len(data).to_bytes(4, 'big') + data)
            await writer.drain()
        except Exception as e:
            self._log.error(f"Failed to send to {addr}: {e}")
            self._remote_connections.pop(addr, None)

    def remote_ref(self, actor_id: str, host: str, port: int) -> ActorRef:
        return ActorRef(actor_id, self, _remote_addr=(host, port))

    async def shutdown(self):
        self._log.info("Shutting down...")
        for actor_id in list(self._actors.keys()):
            await self._stop_actor(actor_id)

        for reader, writer in self._remote_connections.values():
            writer.close()
            await writer.wait_closed()

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        self._running = False
        self._log.info("Shutdown complete")

    async def wait_for_shutdown(self):
        while self._running:
            await asyncio.sleep(0.1)

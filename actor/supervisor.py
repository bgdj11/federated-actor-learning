import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
from actor.actor_system import Actor, ActorRef
from actor.messages import (
    Message, HealthPing, HealthAck, ChildFailed
)

@dataclass
class MonitorChild(Message):
    child_id: str = ""
    actor_class: type = None
    kwargs: dict = None


@dataclass
class GetStatus(Message):
    pass

@dataclass
class StatusReport(Message):
    healthy: list = None
    failed: list = None


class Supervisor(Actor):
    
    def __init__(self, health_check_interval: float = 5.0, health_timeout: float = 3.0):
        super().__init__()
        self.health_check_interval = health_check_interval
        self.health_timeout = health_timeout
        
        self._monitored_children: Dict[str, dict] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._restart_map: Dict[str, tuple] = {}
        self._pending_acks: Dict[str, asyncio.Event] = {}
        
    async def pre_start(self):
        self._health_check_task = asyncio.create_task(self._periodic_health_check())
        self.log.info(f"Supervisor started (check interval: {self.health_check_interval}s)")
        
    async def receive(self, msg: Message):
        if isinstance(msg, MonitorChild):
            await self._add_child(msg)
            
        elif isinstance(msg, HealthAck):
            await self._handle_health_ack(msg)
            
        elif isinstance(msg, ChildFailed):
            await self._handle_child_failed(msg)
            
        elif isinstance(msg, GetStatus):
            await self._handle_get_status(msg)
            
    async def _add_child(self, msg: MonitorChild):
        child_id = msg.child_id
        actor_class = msg.actor_class
        kwargs = msg.kwargs or {}
        
        child_ref = self.context.actor_of(actor_class, child_id, **kwargs)
        
        self._monitored_children[child_id] = {
            "ref": child_ref,
            "status": "starting",
            "last_ack": None,
            "failed_checks": 0
        }
        
        self._restart_map[child_id] = (actor_class, kwargs)
        self._pending_acks[child_id] = asyncio.Event()
        
        self.log.info(f"Monitoring child: {child_id}")
        return child_ref
        
    async def _periodic_health_check(self):
        await asyncio.sleep(1.0)
        
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                for child_id, info in list(self._monitored_children.items()):
                    if info["status"] == "failed":
                        continue
                        
                    child_ref = info["ref"]
                    
                    if child_id not in self._pending_acks:
                        self._pending_acks[child_id] = asyncio.Event()
                    
                    self._pending_acks[child_id].clear()
                    child_ref.tell(HealthPing(), sender=self.context.self_ref)
                    
                    try:
                        await asyncio.wait_for(
                            self._pending_acks[child_id].wait(),
                            timeout=self.health_timeout
                        )
                        info["status"] = "healthy"
                        info["failed_checks"] = 0
                    except asyncio.TimeoutError:
                        info["failed_checks"] += 1
                        self.log.warning(f"Health check timeout for {child_id} (failed: {info['failed_checks']})")
                        if info["failed_checks"] >= 2:
                            await self._restart_child(child_id)
                            
            except Exception as e:
                self.log.error(f"Health check loop error: {e}")
                
    async def _handle_health_ack(self, msg: HealthAck):
        for child_id, info in list(self._monitored_children.items()):
            if info["ref"].actor_id == msg.actor_id:
                info["status"] = "healthy"
                info["last_ack"] = asyncio.get_event_loop().time()
                info["failed_checks"] = 0
                
                if child_id in self._pending_acks:
                    self._pending_acks[child_id].set()  # signalizira da se dogadjaj desio i oslobadja za sve koji cekaju na njega
                break
                
    async def _handle_child_failed(self, msg: ChildFailed):
        child_id = msg.child_id
        if child_id in self._monitored_children:
            self.log.error(f"Child {child_id} failed: {msg.error}")
            await self._restart_child(child_id)
            
    async def _restart_child(self, child_id: str):
        if child_id not in self._monitored_children:
            return
            
        info = self._monitored_children[child_id]
        info["status"] = "failed"
        
        self.log.warning(f"Restarting child: {child_id}")
        
        actor_class, kwargs = self._restart_map[child_id]
        self.context.stop(info["ref"])
        await asyncio.sleep(0.5)
        
        new_ref = self.context.actor_of(actor_class, child_id, **kwargs)
        self._monitored_children[child_id] = {
            "ref": new_ref,
            "status": "starting",
            "last_ack": None,
            "failed_checks": 0
        }
        self._pending_acks[child_id] = asyncio.Event()
        
        self.log.info(f"Child restarted: {child_id}")
        
    async def _handle_get_status(self, msg: GetStatus):
        healthy = [
            cid for cid, info in self._monitored_children.items()
            if info["status"] == "healthy"
        ]
        failed = [
            cid for cid, info in self._monitored_children.items()
            if info["status"] == "failed"
        ]
        
        return StatusReport(healthy=healthy, failed=failed)
        
    async def post_stop(self):
        if self._health_check_task:
            self._health_check_task.cancel()
        self.log.info("Supervisor stopped")

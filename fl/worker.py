import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import Actor, ActorRef
from actor.messages import (
    Message, TrainRequest, ModelUpdate, GlobalModelBroadcast,
    HealthPing, HealthAck, Shutdown
)
from fl.model import SimpleClassifier
from fl.provider import RegisterWorker


class RegionWorker(Actor):

    def __init__(self, region: str, data_dir: str = "dataset", 
                 local_epochs: int = 3, batch_size: int = 32, lr: float = 0.01,
                 provider_ref: ActorRef = None):

        super().__init__()
        self.region = region
        self.data_dir = data_dir
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.model: SimpleClassifier = None
        
        self.provider_ref: ActorRef = provider_ref
        
        self.rounds_completed = 0
        
    async def pre_start(self):
        data_path = os.path.join(self.data_dir, f"region_{self.region}.npz")
        
        if not os.path.exists(data_path):
            self.log.error(f"Dataset not found: {data_path}")
            self.log.info("Run scripts/extract_features.py and scripts/split_regions.py first!")
            return
            
        data = np.load(data_path)
        self.X = data['X']
        self.y = data['y']
        
        self.log.info(f"Loaded region {self.region}: {len(self.y)} samples, "
                      f"labels: {np.unique(self.y)}")
        
        self.model = SimpleClassifier(lr=self.lr)
        self.log.info(f"Model initialized with lr={self.lr}")

        if self.provider_ref:
            advertised_host = "localhost"
            if self.context._system.host not in ("0.0.0.0", "127.0.0.1", "localhost"):
                advertised_host = self.context._system.host

            self.provider_ref.tell(RegisterWorker(
                worker_id=self.actor_id,
                region=self.region,
                host=advertised_host,
                port=self.context._system.port
            ))
            self.log.info("Registered with provider")
        
    async def receive(self, msg: Message):
        if isinstance(msg, TrainRequest):
            await self._handle_train_request(msg)
            
        elif isinstance(msg, GlobalModelBroadcast):
            await self._handle_global_model(msg)
            
        elif isinstance(msg, HealthPing):
            # Dual handling: respond via msg.sender (for local Supervisor) or provider_ref (for remote Provider)
            if msg.sender:
                # Respond to whoever sent the ping (works for local Supervisor)
                msg.sender.tell(HealthAck(actor_id=self.actor_id, status="alive"))
            elif self.provider_ref:
                # Fallback to provider_ref for remote pings
                self.provider_ref.tell(HealthAck(actor_id=self.actor_id, status="alive"))
                
        elif isinstance(msg, Shutdown):
            self.log.info("Shutting down...")
            
    async def _handle_train_request(self, msg: TrainRequest):
        self.log.info(f"Received TrainRequest for round {msg.round_idx}")
        
        if self.X is None:
            self.log.error("No data loaded! Cannot train.")
            return
            
        if msg.global_weights is not None:
            self.model.set_weights(msg.global_weights)
            self.log.info("Applied global weights")
            self.model.set_fedprox(getattr(msg, "mu", 0.0), msg.global_weights)

        self.log.info(f"Starting local training ({self.local_epochs} epochs)...")

        total_loss = 0.0
        total_acc = 0.0
        
        for epoch in range(self.local_epochs):
            metrics = self.model.train_epoch(self.X, self.y, self.batch_size)
            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
            self.log.info(f"  Epoch {epoch+1}/{self.local_epochs}: "
                          f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
        
        avg_loss = total_loss / self.local_epochs
        avg_acc = total_acc / self.local_epochs
        
        update = ModelUpdate(
            worker_id=self.actor_id,
            weights=self.model.get_weights(),
            num_samples=len(self.y),
            metrics={
                'loss': avg_loss,
                'accuracy': avg_acc,
                'region': self.region,
                'round': msg.round_idx
            }
        )
        
        if self.provider_ref:
            self.provider_ref.tell(update)
            self.log.info(f"Sent ModelUpdate to provider "
                          f"(loss={avg_loss:.4f}, acc={avg_acc:.4f})")
        else:
            self.log.warning("No provider_ref set! Update not sent.")
            
        self.rounds_completed += 1
        
    async def _handle_global_model(self, msg: GlobalModelBroadcast):
        self.log.info(f"Received GlobalModelBroadcast for round {msg.round_idx}")
        
        if msg.weights is not None:
            self.model.set_weights(msg.weights)
            self.log.info("Updated local model with global weights")
            
    async def post_stop(self):
        self.log.info(f"Worker {self.region} stopped. "
                      f"Completed {self.rounds_completed} rounds.")

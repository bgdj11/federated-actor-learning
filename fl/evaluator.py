import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actor.actor_system import Actor, ActorRef
from actor.messages import Message, GlobalModelBroadcast, HealthPing, HealthAck, Shutdown
from fl.model import SimpleClassifier
from fl.provider import EvaluationResult, RegisterEvaluator
from storage.persistence import RoundPersistence


class Evaluator(Actor):
    def __init__(self, data_dir: str = "dataset", provider_ref: ActorRef = None):

        super().__init__()
        self.data_dir = data_dir
        self.persistence = RoundPersistence()
        
        self.X_test: np.ndarray = None
        self.y_test: np.ndarray = None
        
        self.model: SimpleClassifier = None
        
        self.provider_ref: ActorRef = provider_ref
        
        self.evaluation_history: list[dict] = []
        
    async def pre_start(self):
        data_path = os.path.join(self.data_dir, "region_D.npz")
        
        if not os.path.exists(data_path):
            self.log.error(f"Test dataset not found: {data_path}")
            self.log.info("Run scripts/extract_features.py and scripts/split_regions.py first!")
            return
            
        data = np.load(data_path)
        self.X_test = data['X']
        self.y_test = data['y']
        
        self.log.info(f"Loaded test data (Region D): {len(self.y_test)} samples, "
                      f"labels: {np.unique(self.y_test)}")
        
        self.model = SimpleClassifier()
        self.log.info("Evaluator ready")

        if self.provider_ref:
            advertised_host = "localhost"
            if self.context._system.host not in ("0.0.0.0", "127.0.0.1", "localhost"):
                advertised_host = self.context._system.host

            self.provider_ref.tell(RegisterEvaluator(
                evaluator_id=self.actor_id,
                host=advertised_host,
                port=self.context._system.port
            ))
            self.log.info("Registered with provider")
        
    async def receive(self, msg: Message):
        
        if isinstance(msg, GlobalModelBroadcast):
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
            self.log.info("Shutting down evaluator...")
            self._print_summary()
            
    async def _handle_global_model(self, msg: GlobalModelBroadcast):
        self.log.info(f"Evaluating model for round {msg.round_idx}...")
        
        if self.X_test is None:
            self.log.error("No test data loaded!")
            return
            
        if msg.weights is None:
            self.log.error("No weights in GlobalModelBroadcast!")
            return
            
        self.model.set_weights(msg.weights)
        
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        self.log.info(f"Round {msg.round_idx} evaluation: "
                      f"accuracy={metrics['accuracy']:.4f}, "
                      f"loss={metrics['loss']:.4f}")
        

        predictions = self.model.predict(self.X_test)
        per_class_acc = {}
        for cls in np.unique(self.y_test):
            mask = self.y_test == cls
            if np.sum(mask) > 0:
                cls_acc = np.mean(predictions[mask] == cls)
                per_class_acc[int(cls)] = float(cls_acc)
                
        self.log.info(f"Per-class accuracy: {per_class_acc}")
        
        self.evaluation_history.append({
            'round': msg.round_idx,
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            'per_class_accuracy': per_class_acc
        })
        
        self.persistence.save_round(
            msg.round_idx,
            eval_metrics={
                "eval_accuracy": metrics['accuracy'],
                "eval_loss": metrics['loss'],
                "per_class_accuracy": per_class_acc
            }
        )
        
        result = EvaluationResult(
            round_idx=msg.round_idx,
            accuracy=metrics['accuracy'],
            loss=metrics['loss']
        )
        
        if self.provider_ref is not None:
            self.provider_ref.tell(result)
            self.log.info("Sent evaluation result to provider")
        else:
            self.log.warning("No provider_ref set!")
            
    def _print_summary(self):
        if not self.evaluation_history:
            self.log.info("No evaluations performed")
            return
            
        self.log.info("\n" + "="*50)
        self.log.info("EVALUATION SUMMARY")
        self.log.info("="*50)
        
        for h in self.evaluation_history:
            self.log.info(f"Round {h['round']}: accuracy={h['accuracy']:.4f}")
            
        if len(self.evaluation_history) > 1:
            first_acc = self.evaluation_history[0]['accuracy']
            last_acc = self.evaluation_history[-1]['accuracy']
            improvement = last_acc - first_acc
            self.log.info(f"\nImprovement: {first_acc:.4f} -> {last_acc:.4f} "
                          f"({improvement:+.4f})")
                          
    async def post_stop(self):
        self._print_summary()
        self.log.info("Evaluator stopped.")

import numpy as np
from typing import Optional


class SimpleClassifier:

    def __init__(self, input_dim: int = 512, num_classes: int = 10, lr: float = 0.01):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr

        self.mu = 0.0
        self.W_global: Optional[np.ndarray] = None
        self.b_global: Optional[np.ndarray] = None
        
        self.W = np.random.randn(input_dim, num_classes) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros(num_classes)
        
    def get_weights(self) -> dict:
        return {
            'W': self.W.copy(),
            'b': self.b.copy()
        }
    
    def set_weights(self, weights: dict):
        self.W = weights['W'].copy()
        self.b = weights['b'].copy()
        
    def softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        # Forward pass: X @ W + b -> softmax -> probabilities
        activations = X @ self.W + self.b
        probs = self.softmax(activations)
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def compute_loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        batch_size = len(y)
        correct_probs = probs[np.arange(batch_size), y]
        loss = -np.mean(np.log(correct_probs + 1e-10))
        return loss
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> dict:
        batch_size = len(y)
        
        probs = self.forward(X)
        loss = self.compute_loss(probs, y)

        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y)
        
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), y] -= 1
        grad_logits /= batch_size

        grad_W = X.T @ grad_logits
        grad_b = np.sum(grad_logits, axis=0)

        if self.mu > 0.0 and self.W_global is not None and self.b_global is not None:
            grad_W += self.mu * (self.W - self.W_global)
            grad_b += self.mu * (self.b - self.b_global)
        
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }
    
    def set_fedprox(self, mu: float, global_weights: Optional[dict] = None):
        self.mu = float(mu)
        if global_weights is None:
            self.W_global = None
            self.b_global = None
        else:
            self.W_global = global_weights['W'].copy()
            self.b_global = global_weights['b'].copy()
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> dict:

        num_samples = len(y)
        indices = np.random.permutation(num_samples)
        
        total_loss = 0.0
        total_correct = 0
        num_batches = 0
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            metrics = self.train_step(X_batch, y_batch)
            
            total_loss += metrics['loss']
            total_correct += metrics['accuracy'] * len(y_batch)
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / num_samples
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        probs = self.forward(X)
        loss = self.compute_loss(probs, y)
        
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }


def federated_averaging(weight_updates: list[tuple[dict, int]]) -> dict:
    if not weight_updates:
        raise ValueError("No weight updates to aggregate!")
    
    total_samples = sum(n for _, n in weight_updates)
    
    first_weights = weight_updates[0][0]
    averaged = {
        'W': np.zeros_like(first_weights['W']),
        'b': np.zeros_like(first_weights['b'])
    }
    
    for weights, num_samples in weight_updates:
        weight_factor = num_samples / total_samples
        averaged['W'] += weight_factor * weights['W']
        averaged['b'] += weight_factor * weights['b']
    
    return averaged

   
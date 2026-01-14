"""
Lokalni FL Test - Sve u jednom procesu

Ovo je pojednostavljen test koji pokreće ceo FL flow lokalno
bez TCP komunikacije. Koristi se za brzo testiranje logike.

Pokretanje:
    python fl/test_local.py
"""

import asyncio
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.model import SimpleClassifier, federated_averaging


async def main():
    print("=" * 60)
    print("LOKALNI FL TEST (bez TCP)")
    print("=" * 60)
    
    # Učitaj podatke za sve regije
    regions = {}
    for region in ['A', 'B', 'C', 'D']:
        data = np.load(f"dataset/region_{region}.npz")
        regions[region] = {'X': data['X'], 'y': data['y']}
        print(f"Region {region}: {len(data['y'])} samples, labels: {np.unique(data['y'])}")
    
    print()
    
    # Kreiraj 3 lokalna modela (simulira workere)
    workers = {
        'A': SimpleClassifier(lr=0.01),
        'B': SimpleClassifier(lr=0.01),
        'C': SimpleClassifier(lr=0.01),
    }
    
    # Globalni model
    global_model = SimpleClassifier()
    
    # FL parametri
    num_rounds = 5
    local_epochs = 3
    
    print("Počinjem Federated Learning...")
    print()
    
    for round_idx in range(1, num_rounds + 1):
        print(f"{'='*60}")
        print(f"RUNDA {round_idx}/{num_rounds}")
        print(f"{'='*60}")
        
        # 1. Pošalji globalne weights svim workerima
        global_weights = global_model.get_weights()
        for worker_id, worker in workers.items():
            worker.set_weights(global_weights)
        
        # 2. Lokalni trening na svakom workeru
        updates = []
        for worker_id in ['A', 'B', 'C']:
            worker = workers[worker_id]
            X = regions[worker_id]['X']
            y = regions[worker_id]['y']
            
            # Treniraj local_epochs
            for epoch in range(local_epochs):
                metrics = worker.train_epoch(X, y, batch_size=32)
            
            print(f"  Worker {worker_id}: loss={metrics['loss']:.4f}, "
                  f"acc={metrics['accuracy']:.4f} ({len(y)} samples)")
            
            # Sačuvaj update
            updates.append((worker.get_weights(), len(y)))
        
        # 3. FedAvg agregacija
        aggregated_weights = federated_averaging(updates)
        global_model.set_weights(aggregated_weights)
        
        # 4. Evaluacija na Region D
        X_test = regions['D']['X']
        y_test = regions['D']['y']
        eval_metrics = global_model.evaluate(X_test, y_test)
        
        print(f"\n  >> Globalni model - Test acc: {eval_metrics['accuracy']:.4f}")
        print()
    
    # Finalni izveštaj
    print("=" * 60)
    print("ZAVRŠENO!")
    print("=" * 60)
    
    final_metrics = global_model.evaluate(regions['D']['X'], regions['D']['y'])
    print(f"Finalna tačnost na Region D: {final_metrics['accuracy']:.4f}")
    
    # Per-class accuracy
    predictions = global_model.predict(regions['D']['X'])
    print("\nPer-class accuracy:")
    for cls in range(10):
        mask = regions['D']['y'] == cls
        if np.sum(mask) > 0:
            cls_acc = np.mean(predictions[mask] == cls)
            print(f"  Class {cls}: {cls_acc:.4f}")


if __name__ == "__main__":
    asyncio.run(main())

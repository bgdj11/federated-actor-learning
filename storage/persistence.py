import sqlite3
import json
import numpy as np
import os
from typing import Optional, Dict, Any
from datetime import datetime


class RoundPersistence:
    
    def __init__(self, db_path: str = "storage/training.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rounds (
                    round_idx INTEGER PRIMARY KEY,
                    weights_blob BLOB,
                    train_metrics TEXT,
                    eval_metrics TEXT,
                    timestamp TEXT
                )
            """)
            conn.commit()
            
    def save_round(self, round_idx: int, weights: Optional[Dict[str, np.ndarray]] = None, 
                   train_metrics: Optional[Dict[str, Any]] = None, 
                   eval_metrics: Optional[Dict[str, Any]] = None):
        train_json = json.dumps(train_metrics) if train_metrics else None
        eval_json = json.dumps(eval_metrics) if eval_metrics else None
        ts = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT weights_blob, train_metrics, eval_metrics FROM rounds WHERE round_idx = ?",
                (round_idx,)
            ).fetchone()
            
            if existing:
                weights_blob = self._serialize_weights(weights) if weights else existing[0]
                train_json = train_json if train_json else existing[1]
                eval_json = eval_json if eval_json else existing[2]
                
                conn.execute("""
                    UPDATE rounds SET weights_blob=?, train_metrics=?, eval_metrics=?, timestamp=?
                    WHERE round_idx = ?
                """, (weights_blob, train_json, eval_json, ts, round_idx))
            else:
                weights_blob = self._serialize_weights(weights) if weights else None
                conn.execute("""
                    INSERT INTO rounds (round_idx, weights_blob, train_metrics, eval_metrics, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (round_idx, weights_blob, train_json, eval_json, ts))
            
            conn.commit()
            
    def load_round(self, round_idx: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT weights_blob, train_metrics, eval_metrics, timestamp
                FROM rounds WHERE round_idx = ?
            """, (round_idx,)).fetchone()
            
        if not row:
            return None
            
        weights_blob, train_json, eval_json, ts = row
        weights = self._deserialize_weights(weights_blob) if weights_blob else None
        train_metrics = json.loads(train_json) if train_json else None
        eval_metrics = json.loads(eval_json) if eval_json else None
        
        return {
            "round_idx": round_idx,
            "weights": weights,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "timestamp": ts
        }
        
    def load_latest_round(self) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT round_idx, weights_blob, train_metrics, eval_metrics, timestamp
                FROM rounds ORDER BY round_idx DESC LIMIT 1
            """).fetchone()
            
        if not row:
            return None
            
        round_idx, weights_blob, train_json, eval_json, ts = row
        weights = self._deserialize_weights(weights_blob) if weights_blob else None
        train_metrics = json.loads(train_json) if train_json else None
        eval_metrics = json.loads(eval_json) if eval_json else None
        
        return {
            "round_idx": round_idx,
            "weights": weights,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "timestamp": ts
        }
        
    def get_all_rounds(self) -> list:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT round_idx, train_metrics, eval_metrics, timestamp
                FROM rounds ORDER BY round_idx
            """).fetchall()
            
        result = []
        for round_idx, train_json, eval_json, ts in rows:
            train_metrics = json.loads(train_json) if train_json else None
            eval_metrics = json.loads(eval_json) if eval_json else None
            result.append({
                "round_idx": round_idx,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "timestamp": ts
            })
        return result
        
    def _serialize_weights(self, weights: Dict[str, np.ndarray]) -> bytes:
        import io
        buf = io.BytesIO()
        np.savez(buf, **weights)
        return buf.getvalue()
        
    def _deserialize_weights(self, blob: bytes) -> Dict[str, np.ndarray]:
        import io
        buf = io.BytesIO(blob)
        data = np.load(buf)
        return {key: data[key] for key in data.files}

import sqlite3
import json
import numpy as np
import os
from typing import Optional, Dict, Any
from datetime import datetime


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
    except Exception:
        pass
    return conn


class RoundPersistence:

    def __init__(self, db_path: str = "storage/training.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_db()

    def _init_db(self):
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rounds (
                    round_idx INTEGER PRIMARY KEY,
                    weights_blob BLOB,
                    train_metrics TEXT,
                    eval_metrics TEXT,
                    timestamp TEXT
                )
                """
            )
            conn.commit()

    def save_round(
        self,
        round_idx: int,
        weights: Optional[Dict[str, np.ndarray]] = None,
        train_metrics: Optional[Dict[str, Any]] = None,
        eval_metrics: Optional[Dict[str, Any]] = None,
    ):
        train_json = json.dumps(train_metrics) if train_metrics else None
        eval_json = json.dumps(eval_metrics) if eval_metrics else None
        ts = datetime.now().isoformat()

        with _connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT weights_blob, train_metrics, eval_metrics FROM rounds WHERE round_idx = ?",
                (round_idx,),
            ).fetchone()

            if existing:
                weights_blob = self._serialize_weights(weights) if weights else existing[0]
                train_json = train_json if train_json else existing[1]
                eval_json = eval_json if eval_json else existing[2]

                conn.execute(
                    """
                    UPDATE rounds SET weights_blob=?, train_metrics=?, eval_metrics=?, timestamp=?
                    WHERE round_idx = ?
                    """,
                    (weights_blob, train_json, eval_json, ts, round_idx),
                )
            else:
                weights_blob = self._serialize_weights(weights) if weights else None
                conn.execute(
                    """
                    INSERT INTO rounds (round_idx, weights_blob, train_metrics, eval_metrics, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (round_idx, weights_blob, train_json, eval_json, ts),
                )

            conn.commit()

    def load_round(self, round_idx: int) -> Optional[Dict[str, Any]]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT weights_blob, train_metrics, eval_metrics, timestamp
                FROM rounds WHERE round_idx = ?
                """,
                (round_idx,),
            ).fetchone()

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
            "timestamp": ts,
        }

    def load_latest_round(self) -> Optional[Dict[str, Any]]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT round_idx, weights_blob, train_metrics, eval_metrics, timestamp
                FROM rounds ORDER BY round_idx DESC LIMIT 1
                """
            ).fetchone()

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
            "timestamp": ts,
        }

    def get_all_rounds(self) -> list:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT round_idx, train_metrics, eval_metrics, timestamp
                FROM rounds ORDER BY round_idx
                """
            ).fetchall()

        result = []
        for round_idx, train_json, eval_json, ts in rows:
            train_metrics = json.loads(train_json) if train_json else None
            eval_metrics = json.loads(eval_json) if eval_json else None
            result.append(
                {
                    "round_idx": round_idx,
                    "train_metrics": train_metrics,
                    "eval_metrics": eval_metrics,
                    "timestamp": ts,
                }
            )
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


class GossipPersistence:
    def __init__(self, db_path: str = "storage/gossip.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_db()

    def _init_db(self):
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS crdt_snapshots (
                    peer_id TEXT,
                    round_num INTEGER,
                    lww_state TEXT,
                    pn_state TEXT,
                    timestamp TEXT,
                    PRIMARY KEY (peer_id, round_num)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS peer_metrics (
                    peer_id TEXT,
                    round_num INTEGER,
                    loss REAL,
                    accuracy REAL,
                    timestamp TEXT,
                    PRIMARY KEY (peer_id, round_num)
                )
                """
            )

            conn.commit()

    def save_crdt_snapshot(
        self,
        peer_id: str,
        round_num: int,
        lww_state: Dict[str, Any],
        pn_state: Dict[str, Any],
    ):
        ts = datetime.now().isoformat()
        lww_json = json.dumps(lww_state)
        pn_json = json.dumps(pn_state)

        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO crdt_snapshots
                (peer_id, round_num, lww_state, pn_state, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (peer_id, int(round_num), lww_json, pn_json, ts),
            )
            conn.commit()

    def load_latest_crdt_snapshot(self, peer_id: str) -> Optional[Dict[str, Any]]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT round_num, lww_state, pn_state, timestamp
                FROM crdt_snapshots
                WHERE peer_id = ?
                ORDER BY round_num DESC
                LIMIT 1
                """,
                (peer_id,),
            ).fetchone()

        if not row:
            return None

        round_num, lww_json, pn_json, ts = row
        return {
            "peer_id": peer_id,
            "round_num": int(round_num),
            "lww_state": json.loads(lww_json),
            "pn_state": json.loads(pn_json),
            "timestamp": ts,
        }

    def get_crdt_snapshot(self, peer_id: str, round_num: int) -> Optional[Dict[str, Any]]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT lww_state, pn_state, timestamp
                FROM crdt_snapshots
                WHERE peer_id = ? AND round_num = ?
                """,
                (peer_id, int(round_num)),
            ).fetchone()

        if not row:
            return None

        lww_json, pn_json, ts = row
        return {
            "peer_id": peer_id,
            "round_num": int(round_num),
            "lww_state": json.loads(lww_json),
            "pn_state": json.loads(pn_json),
            "timestamp": ts,
        }

    def get_all_snapshots_for_round(self, round_num: int) -> list:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT peer_id, lww_state, pn_state, timestamp
                FROM crdt_snapshots
                WHERE round_num = ?
                ORDER BY peer_id
                """,
                (int(round_num),),
            ).fetchall()

        result = []
        for peer_id, lww_json, pn_json, ts in rows:
            result.append(
                {
                    "peer_id": peer_id,
                    "round_num": int(round_num),
                    "lww_state": json.loads(lww_json),
                    "pn_state": json.loads(pn_json),
                    "timestamp": ts,
                }
            )
        return result

    def save_peer_metrics(self, peer_id: str, round_num: int, metrics: Dict[str, Any]):
        ts = datetime.now().isoformat()
        loss = float(metrics.get("loss", 0.0))
        accuracy = float(metrics.get("accuracy", 0.0))

        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO peer_metrics
                (peer_id, round_num, loss, accuracy, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (peer_id, int(round_num), loss, accuracy, ts),
            )
            conn.commit()

    def get_peer_metrics(self, peer_id: str, round_num: int) -> Optional[Dict[str, Any]]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT loss, accuracy, timestamp
                FROM peer_metrics
                WHERE peer_id = ? AND round_num = ?
                """,
                (peer_id, int(round_num)),
            ).fetchone()

        if not row:
            return None

        loss, accuracy, ts = row
        return {
            "peer_id": peer_id,
            "round_num": int(round_num),
            "loss": float(loss),
            "accuracy": float(accuracy),
            "timestamp": ts,
        }

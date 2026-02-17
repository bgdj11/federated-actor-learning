# Federated Learning Platform with Actor Runtime

## Overview
This repository implements a federated learning platform on top of a custom actor system. It supports two execution styles:

1. **Provider mode (coordinated FL)**
   - A central Provider orchestrates rounds.
   - Workers train locally on regional partitions.
   - Aggregator computes weighted model aggregation.
   - Evaluator measures global model quality.

2. **Gossip P2P mode (autonomous FL)**
   - Peers train independently.
   - Peers exchange state through gossip.
   - Each peer recomputes a local copy of the global aggregate.
   - Optional Reporter observes progress.

The codebase is organized to reuse the same actor runtime across both modes.

---

## Repository Layout

- `actor/`
  - `actor_system.py`: actor runtime, local/remote messaging, TCP transport, middleware, SSL support.
  - `messages.py`: shared message definitions.
  - `supervisor.py`: actor-level health checks and child restarts.

- `fl/`
  - `model.py`: simple classifier + federated averaging function.
  - `provider.py`, `worker.py`, `aggregator.py`, `evaluator.py`: coordinated FL pipeline.
  - `gossip_peer.py`: autonomous P2P peer implementation.
  - `gossip_reporter.py`: optional observer for gossip network.
  - `run_scripts/`: process entrypoints for each role.

- `storage/`
  - `persistence.py`: SQLite persistence for provider rounds and gossip snapshots/metrics.

- `scripts/`
  - `extract_features.py`: EuroSAT feature extraction using ResNet18.
  - `split_regions.py`: region partitioning and domain-shift simulation.
  - `generate_certs.py`: helper for SSL test certificates.

- `tests/`
  - Unit and integration tests for runtime, CRDT, gossip, persistence, SSL, middleware, supervision.

- Root scripts
  - `start_provider_mode.ps1`: convenience launcher for coordinated mode.
  - `start_gossip_mode.ps1`: convenience launcher for gossip mode.
  - `cleanup_processes.ps1`: process/port cleanup helper.

---

## Runtime and Dependencies

- Python 3.10+
- PowerShell for launch scripts

Install:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Key dependencies:
- `numpy`, `scikit-learn`, `pandas`
- `pytest`, `pytest-asyncio`, `pytest-cov`
- `torch`, `torchvision` (used by dataset preprocessing scripts)

---

## Actor System Design

The actor runtime (`actor/actor_system.py`) provides:

- **Actor creation and lifecycle**
  - `actor_of(...)` creates a named actor (`actor_id`).
  - `pre_start` / `post_stop` lifecycle hooks.
  - mailbox-based asynchronous processing.

- **Message passing**
  - Local actor-to-actor via mailbox.
  - Remote actor-to-actor via TCP socket.
  - `ActorRef.tell(...)` is fire-and-forget.

- **Transport model**
  - Remote messages are serialized with `pickle`.
  - Framing format: 4-byte big-endian length prefix + payload.
  - Payload stores `(target_actor_id, message)`.
  - Connection reuse/caching per `(host, port)` endpoint.

- **Middleware hooks**
  - Send and receive middleware chains can transform, log, or drop messages.

- **SSL support**
  - Optional TLS server/client contexts through `enable_ssl(...)`.
  - Used by SSL test suite.

### Message model
`actor/messages.py` defines base and domain-specific messages, including:
- FL control messages (`TrainRequest`, `ModelUpdate`, `GlobalModelBroadcast`)
- health and supervision messages (`HealthPing`, `HealthAck`, `ChildFailed`)
- gossip messages (`GossipPeerJoin`, `GossipState`)

---

## Supervision and Recovery

`actor/supervisor.py` monitors child actors in the same process:

- Sends periodic `HealthPing`.
- Expects `HealthAck` within configured timeout.
- Restarts child actors after repeated health check failures.

Where supervision is used:
- Provider process can supervise Provider actor.
- Aggregator, Evaluator, Workers can run under local Supervisor.
- Gossip peer process can run its peer actor under local Supervisor.

Important scope note:
- This is **actor-level in-process recovery**.
- It does not restart a crashed Python process. Process-level recovery requires external process management.

---

## Data Pipeline

### Source dataset
Preprocessing scripts use **EuroSAT** image dataset.

### Feature extraction (`scripts/extract_features.py`)
- Loads pretrained `ResNet18` (`IMAGENET1K_V1`) from torchvision.
- Removes final FC layer.
- Extracts 512-dimensional feature vectors.
- Saves `eurosat_features.npz` with:
  - `features`
  - `labels`

### Regional split and domain shift (`scripts/split_regions.py`)
The extracted features are partitioned into regions and transformed:

- Region A: rural classes + haze/noise shift
- Region B: urban classes + color shift
- Region C: water/agro classes + contrast shift
- Region D: control region (all classes, no shift)

Output files:
- `dataset/region_A.npz`
- `dataset/region_B.npz`
- `dataset/region_C.npz`
- `dataset/region_D.npz`

Each region file contains:
- `X` (feature matrix)
- `y` (labels)

---

## Model and Training

`fl/model.py` defines `SimpleClassifier`:
- Linear classifier over feature vectors (`X @ W + b`) with softmax.
- Cross-entropy loss.
- SGD updates.
- Batch-level and epoch-level training helpers.

### Federated aggregation
`federated_averaging(...)` computes weighted average by sample counts:
- Input: list of `(weights_dict, num_samples)`.
- Output: aggregated `W`, `b`.

### FedProx support in coordinated mode
`TrainRequest` includes `mu`.
If `mu > 0`, worker applies proximal term toward received global weights.

---

## Provider Mode (Coordinated FL)

### Roles
- **Provider** (`fl/provider.py`)
  - orchestrates rounds
  - tracks registrations
  - dispatches train requests
  - receives aggregated results and evaluation

- **Workers** (`fl/worker.py`)
  - load regional data (`region_A/B/C.npz`)
  - train local model for configured local epochs
  - send model updates to Provider

- **Aggregator** (`fl/aggregator.py`)
  - performs weighted FedAvg
  - computes training summary metrics
  - persists round state to `training.db`

- **Evaluator** (`fl/evaluator.py`)
  - uses `region_D.npz` as evaluation set
  - evaluates global model per round
  - persists eval metrics to `training.db`

### Execution
Convenience launcher:

```powershell
.\start_provider_mode.ps1
```

Or run components manually via `fl/run_scripts/`:
- `run_provider.py`
- `run_aggregator.py`
- `run_evaluator.py`
- `run_workers.py`

---

## Gossip P2P Mode (Autonomous FL)

### Core behavior (`fl/gossip_peer.py`)
Each peer:

1. Loads/recovers CRDT snapshot (if available).
2. Loads regional dataset.
3. Publishes local model into CRDT under key `model/<peer_id>`.
4. Runs two background loops:
   - local training loop
   - gossip exchange loop

### Membership and topology discovery
- Initial contacts are provided via seed endpoints.
- `GossipPeerJoin` bootstraps direct neighbor awareness.
- `peer_info` piggybacked inside gossip messages spreads membership knowledge.

### Global model in P2P mode
There is no central global model owner.
Each peer computes a local copy of a global aggregate by:
- collecting known peer model entries from LWW map,
- applying weighted FedAvg,
- applying the aggregate locally.

### Reporter (`fl/gossip_reporter.py`)
Reporter is optional and observer-only:
- receives periodic `GossipState` summaries from peers,
- tracks latest round/delta per peer,
- prints periodic network reports.

### Execution
Convenience launcher:

```powershell
.\start_gossip_mode.ps1
```

Manual entrypoints:
- `fl/run_scripts/run_gossip_peer.py`
- `fl/run_scripts/run_gossip_reporter.py`

---

## CRDT Layer

Implemented in `fl/crdt.py`.

### LWWMap
- Stores key/value entries with per-entry timestamps.
- Supports both:
  - op-based merge (`put/delete` deltas)
  - state-based merge (`merge_state`) used by gossip snapshots.

### PNCounter
- Standard positive/negative counter CRDT.
- Merge takes component-wise maxima.

### Timestamp semantics
- Lamport-style ordering by `(logical_time, replica_id)`.
- Merge paths advance local logical clock using receive semantics.

### Tombstones
- Used for delete semantics in LWWMap.
- Keep delete-wins behavior under reordering/late delivery.
- In current gossip FL flow, model keys are mostly updated (put), so tombstones are not central to normal operation.

---

## Persistence

`storage/persistence.py` contains two persistence components.

### RoundPersistence (`storage/training.db`)
Used by coordinated mode:
- round weights
- train metrics
- eval metrics

### GossipPersistence (`storage/gossip.db`)
Used by gossip mode:
- CRDT snapshots (`crdt_snapshots`)
- per-peer metrics (`peer_metrics`)

### SQLite WAL mode
Connections enable WAL (`journal_mode=WAL`) for better concurrent read/write behavior.
You may see auxiliary files (`.db-wal`, `.db-shm`) while databases are active.

---

## Networking, TCP, and SSL

### TCP transport
- Actor servers listen with asyncio TCP server.
- Remote actor refs are addressed by `(host, port, actor_id)`.
- Messages are serialized and length-framed.

### SSL/TLS support
- `ActorSystem.enable_ssl(certfile, keyfile, ca_cert=None)` enables TLS server and client contexts.
- If `ca_cert` is omitted, client verification is relaxed in this project setup.
- `scripts/generate_certs.py` can generate self-signed certs for local testing.

---

## Test Coverage Map

Main test modules:

- `test_actor_basics.py`
  - actor creation, behavior switching, parent/child lifecycle

- `test_remote_messaging.py`
  - remote refs, bidirectional messaging, connection caching

- `test_ssl_encryption.py`
  - SSL context setup, encrypted messaging, compatibility checks

- `test_middleware.py`
  - send/receive middleware chains, filtering, transformations

- `test_supervisor.py`
  - child monitoring and restart behavior

- `test_persistence.py`
  - round persistence I/O behavior

- `test_crdt.py`
  - timestamp ordering, merge behavior for LWWMap and PNCounter

- `test_gossip.py`
  - autonomous peer behavior, merge and convergence checks, reporter behavior

- `test_gossip_integration.py`
  - multi-system gossip exchange over real TCP path

Run all tests:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

---

## Typical Workflows

### 1) Prepare dataset pipeline

```powershell
.\venv\Scripts\Activate.ps1
python scripts\extract_features.py
python scripts\split_regions.py
```

### 2) Run coordinated mode

```powershell
.\start_provider_mode.ps1
```

### 3) Run gossip mode

```powershell
.\start_gossip_mode.ps1
```

### 4) Cleanup between runs

```powershell
.\cleanup_processes.ps1
```

---

## Operational Notes

- The actor runtime uses in-memory mailboxes and asynchronous task loops.
- Gossip mode is eventually consistent by design; peers can be temporarily out of sync.
- Reporter output lag is expected: reports are periodic snapshots, not per-message tracing.
- Supervision handles actor failures inside a running process; process crashes require external management if automatic process restart is needed.

---

## Current Scope and Limitations

This project currently provides:
- custom actor framework with remote messaging
- optional TLS transport
- coordinated FL pipeline
- autonomous gossip FL pipeline
- CRDT-based state exchange with persistence
- test suite covering runtime and protocol-level behavior

Not included by default:
- external process supervisor for OS-level crash restart
- production-grade authentication/authorization model
- centralized experiment tracking service

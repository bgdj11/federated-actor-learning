import pytest
import asyncio
from storage.persistence import RoundPersistence
import numpy as np


@pytest.fixture
def persistence(tmp_path):
    db_file = tmp_path / "test_training.db"
    p = RoundPersistence(db_path=str(db_file))
    yield p


def test_persistence_save_load(persistence):
    weights = {
        'W': np.random.randn(512, 10),
        'b': np.zeros(10)
    }
    metrics = {
        'train_avg_loss': 0.5,
        'train_avg_accuracy': 0.85
    }
    
    persistence.save_round(1, weights=weights, train_metrics=metrics)
    
    loaded = persistence.load_round(1)
    assert loaded is not None
    assert loaded['round_idx'] == 1
    assert np.allclose(loaded['weights']['W'], weights['W'])
    assert loaded['train_metrics'] == metrics


def test_persistence_save_train_and_eval(persistence):
    weights = {
        'W': np.random.randn(512, 10),
        'b': np.zeros(10)
    }
    train_metrics = {
        'train_avg_loss': 0.5,
        'train_avg_accuracy': 0.85
    }
    eval_metrics = {
        'eval_accuracy': 0.82,
        'eval_loss': 0.6
    }
    
    persistence.save_round(1, weights=weights, train_metrics=train_metrics)
    persistence.save_round(1, eval_metrics=eval_metrics)
    
    loaded = persistence.load_round(1)
    assert loaded['train_metrics'] == train_metrics
    assert loaded['eval_metrics'] == eval_metrics


def test_persistence_multiple_rounds(persistence):
    for r in range(1, 4):
        weights = {'W': np.random.randn(512, 10), 'b': np.zeros(10)}
        metrics = {'train_avg_loss': 1.0 / r, 'train_avg_accuracy': 0.7 + 0.05*r}
        persistence.save_round(r, weights=weights, train_metrics=metrics)
    
    rounds = persistence.get_all_rounds()
    assert len(rounds) == 3
    assert rounds[0]['round_idx'] == 1
    assert rounds[2]['round_idx'] == 3


def test_persistence_load_latest(persistence):
    for r in range(1, 4):
        weights = {'W': np.random.randn(512, 10), 'b': np.zeros(10)}
        metrics = {'train_avg_loss': 1.0 / r, 'train_avg_accuracy': 0.7 + 0.05*r}
        persistence.save_round(r, weights=weights, train_metrics=metrics)
    
    latest = persistence.load_latest_round()
    assert latest['round_idx'] == 3
    assert latest['train_metrics']['train_avg_accuracy'] == 0.85


def test_persistence_no_weights(persistence):
    metrics = {'train_avg_loss': 0.5}
    persistence.save_round(1, train_metrics=metrics)
    
    loaded = persistence.load_round(1)
    assert loaded['weights'] is None
    assert loaded['train_metrics'] == metrics


def test_persistence_load_nonexistent(persistence):
    loaded = persistence.load_round(999)
    assert loaded is None

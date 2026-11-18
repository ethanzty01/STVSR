"""Check balanced sampling and durable optimizer/RNG continuation."""
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_tiny import (
    BalancedStepSampler, SeededMixedDataset, atomic_checkpoint, evaluate_after_training,
    restore_rng, rng_state,
)


class _RandomDataset:
    def __init__(self, source, length):
        self.source, self.length = source, length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.source, index, random.random()


def test_balanced_sampler_resume_replays_same_samples_and_augmentations():
    complete = list(BalancedStepSampler([7, 3], 8, 12, seed=77))
    resumed = list(BalancedStepSampler([7, 3], 8, 12, seed=77, start_step=5))
    assert resumed == complete[5:]
    for batch in complete:
        assert len(batch) == 8
        assert sum(source == 0 for source, _, _ in batch) == 4
        assert all(0 <= index < [7, 3][source] for source, index, _ in batch)
    dataset = SeededMixedDataset([_RandomDataset(0, 7), _RandomDataset(1, 3)])
    expected = [[dataset[key] for key in batch] for batch in complete[5:]]
    random.seed(999)
    actual = [[dataset[key] for key in batch] for batch in resumed]
    assert actual == expected
    assert len(BalancedStepSampler([7, 3], 8, 12, 77, start_step=12)) == 0


class _SmallStateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.dropout = torch.nn.Dropout(.5)
        self.embed_dim, self.depth = 32, 2
        self.drop_path_rate, self.head = 0., 'direct'

    def forward(self, x):
        return self.linear(self.dropout(x))


def _components():
    model = _SmallStateModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    return model, optimizer, scheduler, scaler


def _step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    # The dropout and generated batch exercise torch RNG restoration.
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach().clone()


def test_checkpoint_restores_exact_next_optimizer_step_and_random_streams(tmp_path):
    before = rng_state()
    try:
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        model, optimizer, scheduler, scaler = _components()
        for _ in range(2):
            _step(model, optimizer, scheduler)
        path = tmp_path / 'last.pth'
        atomic_checkpoint(path, model, step=2, base_step=20000,
                          validation={'vimeo': 31., 'reds': 29., 'balanced_mean': 30.},
                          best_score=30., optimizer=optimizer, scheduler=scheduler,
                          scaler=scaler, elapsed_seconds=12.5)
        expected_random = (random.random(), np.random.random())
        expected_loss = _step(model, optimizer, scheduler)
        expected_weights = {name: value.clone() for name, value in model.state_dict().items()}
        expected_optimizer = optimizer.state_dict()

        restored = torch.load(path, map_location='cpu', weights_only=True)
        assert restored['step'] == 2
        assert restored['cumulative_step'] == 20002
        assert restored['elapsed_seconds'] == 12.5
        assert not path.with_suffix('.pth.tmp').exists()
        resumed, resumed_optimizer, resumed_scheduler, resumed_scaler = _components()
        resumed.load_state_dict(restored['state_dict'])
        resumed_optimizer.load_state_dict(restored['optimizer'])
        resumed_scheduler.load_state_dict(restored['scheduler'])
        resumed_scaler.load_state_dict(restored['scaler'])
        restore_rng(restored['rng'])
        assert (random.random(), np.random.random()) == expected_random
        actual_loss = _step(resumed, resumed_optimizer, resumed_scheduler)
        torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
        for name, value in resumed.state_dict().items():
            torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)
        assert resumed_scheduler.state_dict() == scheduler.state_dict()
        actual_optimizer = resumed_optimizer.state_dict()
        assert actual_optimizer['param_groups'] == expected_optimizer['param_groups']
        for parameter, state in actual_optimizer['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, expected_optimizer['state'][parameter][name], atol=0, rtol=0)
    finally:
        restore_rng(before)


def test_automatic_evaluation_reuses_process_after_cleanup(monkeypatch, tmp_path):
    import evaluate_tiny_mixed
    import train_tiny
    events = []
    arguments = []
    monkeypatch.setattr(torch.compiler, 'reset', lambda: events.append('release_graphs'))
    monkeypatch.setattr(train_tiny.gc, 'collect', lambda: events.append('collect'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: events.append('release_cache'))

    def evaluate(argv):
        events.append('evaluate')
        arguments.extend(argv)

    monkeypatch.setattr(evaluate_tiny_mixed, 'main', evaluate)
    args = SimpleNamespace(compile=True, device='cuda:0', output=tmp_path / 'run with spaces',
                           data_root=tmp_path / 'data', teacher=tmp_path / 'teacher.pth',
                           vimeo_batch_size=4, reds_batch_size=1, threads=2)
    evaluate_after_training(args)
    assert events == ['release_graphs', 'collect', 'release_cache', 'evaluate']
    options = dict(zip(arguments[::2], arguments[1::2]))
    assert options == {'--run-dir': str(args.output), '--data-root': str(args.data_root),
                       '--teacher': str(args.teacher), '--vimeo-batch-size': '4',
                       '--reds-batch-size': '1', '--device': 'cuda:0', '--threads': '2'}

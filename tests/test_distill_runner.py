"""Check KD time alignment and loss/DropPath gradient behavior."""
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_tiny import aggregate, baseline, charbonnier, score_batch, teacher_targets
from distill_metrics import compute_metrics
from models.torch_layers import DropPath


class DistillationRunnerTest(unittest.TestCase):
    def test_teacher_targets_align_each_student_pair_to_three_teacher_frames(self):
        # Each fake output pixel encodes its batch index and HR timestamp.
        output = (100 * torch.arange(3)[:, None] + torch.arange(7)[None]).float()
        output = output[:, :, None, None, None].expand(3, 7, 3, 2, 2)
        teacher_input = torch.rand(3, 4, 3, 1, 1)

        def teacher(x):
            torch.testing.assert_close(x, teacher_input)
            return output

        batch = {'teacher_lr': teacher_input, 'pair_index': torch.tensor([0, 1, 2])}
        actual = teacher_targets(teacher, batch, torch.device('cpu'))
        expected = torch.stack([output[0, 0:3], output[1, 2:5], output[2, 4:7]])
        self.assertEqual(tuple(actual.shape), (3, 3, 3, 2, 2))
        torch.testing.assert_close(actual, expected)

    def test_charbonnier_has_finite_gradients_with_correct_direction(self):
        prediction = torch.tensor([-2., 0., 3.], requires_grad=True)
        target = torch.zeros_like(prediction)
        loss = charbonnier(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertLess(prediction.grad[0].item(), 0.)
        self.assertEqual(prediction.grad[1].item(), 0.)
        self.assertGreater(prediction.grad[2].item(), 0.)
        self.assertAlmostEqual(charbonnier(target, target).item(), .001, places=7)

    def test_baseline_preserves_endpoints_and_midpoint(self):
        x = torch.stack([torch.zeros(3, 2, 4), torch.ones(3, 2, 4)], dim=0)[None]
        output = baseline(x)
        self.assertEqual(tuple(output.shape), (1, 3, 3, 8, 16))
        for i, value in enumerate([0., .5, 1.]):
            torch.testing.assert_close(output[:, i], torch.full_like(output[:, i], value))

    def test_drop_path_evaluation_and_zero_probability_are_identity(self):
        x = torch.randn(8, 2, 3, 4, 5)
        self.assertIs(DropPath(.7).eval()(x), x)
        self.assertIs(DropPath(0.).train()(x), x)

    def test_drop_path_uses_one_mask_per_batch_sample(self):
        torch.manual_seed(123)
        x = torch.ones(64, 2, 3, 4, 5, requires_grad=True)
        output = DropPath(.5).train()(x)
        per_sample = output[:, 0, 0, 0, 0]
        self.assertEqual(set(per_sample.detach().tolist()), {0., 2.})
        torch.testing.assert_close(output, per_sample[:, None, None, None, None].expand_as(x))
        output.sum().backward()
        torch.testing.assert_close(x.grad, output.detach())

    def test_score_batch_preserves_model_clip_frame_labels_and_order(self):
        generator = torch.Generator().manual_seed(73)
        target = torch.rand(2, 3, 3, 16, 20, generator=generator) * .6 + .1
        predictions = {'teacher': target + .02, 'distilled': target + .04}
        keys, groups = ['clip_z', 'clip_a'], ['fast', 'slow']
        rows = score_batch(predictions, target, keys, groups)
        expected_labels = [(model, keys[b], groups[b], frame)
                           for model in predictions for b in range(2) for frame in range(3)]
        actual_labels = [(row['model'], row['key'], row['group'], row['frame']) for row in rows]
        self.assertEqual(actual_labels, expected_labels)
        for name, prediction in predictions.items():
            direct = compute_metrics(prediction, target)['per_frame']
            for b in range(2):
                for frame in range(3):
                    row = next(row for row in rows if row['model'] == name
                               and row['key'] == keys[b] and row['frame'] == frame)
                    for metric, value in direct[b][frame].items():
                        self.assertAlmostEqual(row[metric], value, places=12)

    def test_parallel_batch_aggregation_matches_direct_group_and_time_metrics(self):
        generator = torch.Generator().manual_seed(19)
        target = torch.rand(4, 3, 3, 16, 20, generator=generator) * .6 + .1
        # Distinct errors expose swapped batch, temporal, or model dimensions.
        error = (torch.arange(12).reshape(4, 3, 1, 1, 1) + 1) / 255
        predictions = {'supervised': target + error, 'distilled': target + 2 * error}
        keys = ['c3', 'c1', 'c2', 'c0']
        groups = ['slow', 'medium', 'slow', 'fast']
        # Include an uneven final batch, as the full evaluation uses 7,815 clips.
        slices = [slice(0, 3), slice(3, 4)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(score_batch, {k: v[s] for k, v in predictions.items()},
                                   target[s], keys[s], groups[s]) for s in slices]
            rows = [row for future in futures for row in future.result()]
        measured = aggregate(rows)
        for model, prediction in predictions.items():
            for group in ['all', 'slow', 'medium', 'fast']:
                indices = [i for i, label in enumerate(groups) if group == 'all' or label == group]
                direct = compute_metrics(prediction[indices], target[indices])
                for kind in ['all', 'endpoints', 'center']:
                    for metric, value in direct[kind].items():
                        self.assertAlmostEqual(measured[model][group][kind][metric], value, places=12)


if __name__ == '__main__':
    unittest.main()

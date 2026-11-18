"""Training checks for progressive surgery, freezing, selection and export."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from distill_metrics import compute_metrics
from models.RSTT_progressive import ProgressiveRSTT
from models.RSTT_tiny import RSTTTiny
from train_tiny import (
    STAGES, atomic_save, checkpoint, export_final, freeze_for_step, gradient_coverage,
    normalized_feature_loss, optimizer_for_stage, parse_progressive_args,
    progressive_train_step, restore_rng, take_temporal, triplet, validate_model,
)


class ProgressiveTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def small_model(self, frames=2):
        return ProgressiveRSTT(embed_dim=32, depths=[2] * 8, num_frames=frames,
                               head='direct', baseline_mode='separable', drop_path_rate=0.)

    def training_args(self):
        return SimpleNamespace(compile=False, precision='fp32', head_lr=1e-4,
                               backbone_lr=2e-5, width_lr=5e-5, kd_weight=.3,
                               anchor_weight=.1, feature_weight=.05)

    def batch(self):
        teacher_lr = torch.rand(2, 4, 3, 32, 32)
        pair = torch.tensor([0, 2])
        return {'teacher_lr': teacher_lr, 'student_lr': take_temporal(teacher_lr, pair, 2, 1),
                'pair_index': pair, 'gt': torch.rand(2, 3, 3, 128, 128)}

    def test_temporal_and_channel_feature_correspondence(self):
        pairs = torch.tensor([0, 2])
        channels = torch.tensor([0, 2, 4])
        teacher = {'encoder': [torch.randn(2, 4, 5, 4, 4) for _ in range(4)],
                   'decoder': [torch.randn(2, 7, 5, 4, 4) for _ in range(4)]}
        student = {key: [take_temporal(value, pairs, 2 if key == 'encoder' else 3,
                                     1 if key == 'encoder' else 2).index_select(2, channels)
                         for value in values] for key, values in teacher.items()}
        self.assertEqual(normalized_feature_loss(student, teacher, pairs, channels).item(), 0.)
        student['encoder'][0] = student['encoder'][0] + .5
        self.assertGreater(normalized_feature_loss(student, teacher, pairs, channels).item(), 0.)

    def test_head_freeze_then_unfreeze_updates_backbone(self):
        student = self.small_model()
        previous = copy.deepcopy(student).eval().requires_grad_(False)
        original = self.small_model(4).eval().requires_grad_(False)
        args = self.training_args()
        stage = {'name': '01_direct_head', 'steps': 4, 'freeze_steps': 1}
        optimizer, _, scaler = optimizer_for_stage(student, stage, torch.device('cpu'), args)
        batch = self.batch()
        original_parameter = previous.input_proj.proj[0].weight.clone()
        initial_backbone = student.input_proj.proj[0].weight.clone()
        initial_head = student.conv_last.weight.clone()
        freeze_for_step(student, stage, 0)
        progressive_train_step(args, batch, student, previous, original, student.get_config(), previous.get_config(),
                   optimizer, scaler, torch.device('cpu'), torch.arange(32))
        torch.testing.assert_close(student.input_proj.proj[0].weight, initial_backbone, atol=0, rtol=0)
        self.assertFalse(torch.equal(student.conv_last.weight, initial_head))
        freeze_for_step(student, stage, 1)
        losses = progressive_train_step(args, batch, student, previous, original, student.get_config(), previous.get_config(),
                            optimizer, scaler, torch.device('cpu'), torch.arange(32), check_gradients=True)
        self.assertFalse(torch.equal(student.input_proj.proj[0].weight, initial_backbone))
        self.assertEqual(losses['gradient_coverage']['parameter_tensors_with_gradients'],
                         len(list(student.parameters())))
        torch.testing.assert_close(previous.input_proj.proj[0].weight, original_parameter, atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in previous.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in original.parameters()))

    def test_gradient_guard_detects_unused_trainable_parameters(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        model[1].weight.grad = torch.zeros_like(model[1].weight)
        model[1].bias.grad = torch.zeros_like(model[1].bias)
        with self.assertRaisesRegex(RuntimeError, 'Missing gradients for 2/4'):
            gradient_coverage(model)

    def test_validation_matches_existing_metric_protocol(self):
        for frames in (2, 4):
            model, batch = self.small_model(frames).eval(), self.batch()
            inputs = batch['teacher_lr' if frames == 4 else 'student_lr']
            with torch.no_grad():
                output = triplet(model(inputs), batch)
            metrics = compute_metrics(output, batch['gt'])
            expected = np.mean([row['psnr_y'] for clip in metrics['per_frame'] for row in clip])
            actual = validate_model(model, {'vimeo': [batch], 'reds': [batch]}, torch.device('cpu'))
            self.assertAlmostEqual(actual['balanced_mean'], expected, places=10)

    def test_optimizer_checkpoint_resumes_same_parameter_update(self):
        args, stage = self.training_args(), {'name': '08_gt_finetune', 'steps': 4}
        model, batch = self.small_model(), self.batch()
        optimizer, scheduler, scaler = optimizer_for_stage(model, stage, torch.device('cpu'), args)

        def update(student, optimizer, scheduler, scaler):
            progressive_train_step(args, batch, student, None, None, student.get_config(), student.get_config(),
                       optimizer, scaler, torch.device('cpu'), None, supervised=True)
            scheduler.step()

        update(model, optimizer, scheduler, scaler)
        stream = io.BytesIO()
        torch.save(checkpoint(model, stage, 1, {}, 0., {}, 0., 0, optimizer, scheduler, scaler), stream)
        update(model, optimizer, scheduler, scaler)
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
        resumed = ProgressiveRSTT(**state['model_config'])
        resumed.load_state_dict(state['state_dict'])
        other_optimizer, other_scheduler, other_scaler = optimizer_for_stage(resumed, stage, torch.device('cpu'), args)
        other_optimizer.load_state_dict(state['optimizer'])
        other_scheduler.load_state_dict(state['scheduler'])
        other_scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])
        update(resumed, other_optimizer, other_scheduler, other_scaler)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(scheduler.state_dict(), other_scheduler.state_dict())

    def test_stage_schedule_and_final_export_choose_validation_best(self):
        self.assertEqual(sum(stage['steps'] for stage in STAGES), 140000)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, stages = parse_progressive_args(['--output', str(root), '--step-scale', '.0001', '--no-compile'])
            self.assertEqual([stage['steps'] for stage in stages], [1, 2, 1, 1, 2, 2, 3, 2])
            model = self.small_model()
            summaries = []
            for name, score in (('07_width32', 30.), ('08_gt_finetune', 29.)):
                path = root / f'{name}.pth'
                state = checkpoint(model, {'name': name}, 2, {'balanced_mean': score}, score, {}, 1., 100)
                atomic_save(path, state)
                summaries.append({'stage': name, 'model_config': model.get_config(),
                                  'recovered_best': {'balanced_mean': score}, 'checkpoint': str(path)})
            final = export_final(args, summaries)
            self.assertEqual(final['selected_stage'], '07_width32')
            loaded = torch.load(root / 'final.pth', map_location='cpu', weights_only=True)
            tiny = RSTTTiny(**loaded['model_config'])
            tiny.load_state_dict(loaded['state_dict'], strict=True)
            self.assertEqual(loaded['parameters'], 286140)
            self.assertNotIn('optimizer', loaded)
            self.assertEqual(json.loads((root / 'final_manifest.json').read_text())['sha256'], final['sha256'])


if __name__ == '__main__':
    unittest.main()

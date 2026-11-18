"""Numerical and transfer checks for the staged RSTT experiment."""

import json
from pathlib import Path
import unittest

import torch

from models.RSTT import RSTT
from models.RSTT_progressive import (ProgressiveRSTT, load_original_teacher,
                                     migrate_model, select_channels)
from models.RSTT_tiny import RSTTTiny


class ProgressiveModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)

    def test_original_teacher_numerical_equivalence(self):
        teacher = RSTT(depths=[4] * 8, drop_path_rate=0).eval()
        progressive = ProgressiveRSTT(drop_path_rate=0).eval()
        report = load_original_teacher(progressive, teacher)
        self.assertEqual(len(report["ignored_unused_parameters"]), 2)
        x = torch.rand(1, 4, 3, 32, 40)
        with torch.no_grad():
            expected = teacher(x)
            actual, features = progressive(x, return_features=True)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual([f.shape[-2:] for f in features["encoder"]],
                         [(32, 40), (16, 20), (8, 10), (4, 5)])
        self.assertEqual([f.shape[1] for f in features["decoder"]], [7] * 4)

    def test_final_tiny_strict_loading_and_numerical_equivalence(self):
        progressive = ProgressiveRSTT(embed_dim=32, depths=[2] * 8, num_frames=2,
                                      head="direct", baseline_mode="separable",
                                      drop_path_rate=0).eval()
        tiny = RSTTTiny(embed_dim=32, depth=2, head="direct", drop_path_rate=0).eval()
        tiny.load_state_dict(progressive.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in tiny.parameters()), 286140)
        for shape in ((2, 2, 3, 32, 40), (1, 2, 3, 19, 23)):
            x = torch.rand(shape)
            with torch.no_grad():
                torch.testing.assert_close(progressive(x), tiny(x), atol=0, rtol=0)

    def test_saved_teacher_batch_numerical_equivalence(self):
        path = Path(__file__).resolve().parents[1] / "checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth"
        if not path.exists():
            self.skipTest("Saved RSTT-S checkpoint is required for the integration check")
        teacher = RSTT(depths=[4] * 8, drop_path_rate=0).eval()
        teacher.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        progressive = ProgressiveRSTT.from_original(path).eval()
        x = torch.rand(2, 4, 3, 32, 40)
        with torch.no_grad():
            expected, actual = teacher(x), progressive(x)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

    def test_head_replacement_preserves_original_baseline(self):
        source = ProgressiveRSTT(embed_dim=32, depths=[2] * 8, drop_path_rate=0).eval()
        target, report = migrate_model(source, {"head": "direct"})
        self.assertEqual(target.baseline_mode, "original")
        self.assertEqual(report["fresh_parameters"], ["conv_last.weight", "conv_last.bias"])
        x = torch.rand(1, 4, 3, 32, 32)
        with torch.no_grad():
            expected = torch.nn.functional.interpolate(x.permute(0, 2, 1, 3, 4),
                       size=(7, 128, 128), mode="trilinear", align_corners=False).permute(0, 2, 1, 3, 4)
            torch.testing.assert_close(target(x), expected, atol=0, rtol=0)
        json.dumps(report)

    def test_temporal_bias_crops_preserve_signed_offsets(self):
        source = ProgressiveRSTT(embed_dim=32, depths=[2] * 8, head="direct", drop_path_rate=0)
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if name.endswith("relative_position_bias_table"):
                    parameter.copy_(torch.arange(parameter.numel()).reshape_as(parameter))
        target, report = migrate_model(source, {"num_frames": 2, "baseline_mode": "separable"})
        for key, crop in report["temporal_bias_crops"].items():
            old = source.state_dict()[key]
            new = target.state_dict()[key]
            start = crop["source_offset_start"]
            bins = crop["target_temporal_bins"]
            expected = old.reshape(-1, 49, old.shape[1])[start:start + bins].reshape_as(new)
            torch.testing.assert_close(new, expected, atol=0, rtol=0)
            attention = target.get_submodule(key.removesuffix(".relative_position_bias_table"))
            index = attention.relative_position_index
            # Same spatial position and time maps to temporal offset zero.
            self.assertEqual(index[0, 0].item(), (bins // 2) * 49 + 24)
            self.assertEqual(index.shape[0], attention.num_frames_q * 16)
            self.assertEqual(index.shape[1], attention.num_frames_kv * 16)
        self.assertEqual(len(report["temporal_bias_crops"]), 24)

    def test_depth_selection_retains_window_pair(self):
        source = ProgressiveRSTT(embed_dim=32, depths=[4] * 8, num_frames=2,
                                 head="direct", baseline_mode="separable", drop_path_rate=0)
        pairs = {f"{side}_layers.{i}": [2, 3]
                 for side in ("encoder", "decoder") for i in range(4)}
        target, _ = migrate_model(source, {"depths": [2] * 8}, block_pairs=pairs)
        for side in ("encoder", "decoder"):
            for i in range(4):
                old = getattr(source, f"{side}_layers")[i].blocks
                new = getattr(target, f"{side}_layers")[i].blocks
                for j in range(2):
                    torch.testing.assert_close(new[j].norm1.weight, old[j + 2].norm1.weight)
                    attention = "attn" if side == "encoder" else "attn1"
                    torch.testing.assert_close(getattr(new[j], attention).q.weight,
                                               getattr(old[j + 2], attention).q.weight)

    def test_width_pruning_and_gradients(self):
        source = ProgressiveRSTT(embed_dim=96, depths=[2] * 8, num_frames=2,
                                 head="direct", baseline_mode="separable", drop_path_rate=0)
        x = torch.rand(1, 2, 3, 32, 32)
        for width in (64, 48, 32):
            channels = select_channels(source, [x], width)
            for group in range(16):
                self.assertEqual(sum(i // (source.embed_dim // 16) == group for i in channels), width // 16)
            target, report = migrate_model(source, {"embed_dim": width}, channel_indices=channels)
            indices = torch.tensor(channels)
            old_q = source.encoder_layers[0].blocks[0].attn.q.weight
            torch.testing.assert_close(target.encoder_layers[0].blocks[0].attn.q.weight,
                                       old_q[indices][:, indices])
            old_kv = source.encoder_layers[0].blocks[0].attn.kv.weight
            kv_indices = torch.cat((indices, indices + source.embed_dim))
            torch.testing.assert_close(target.encoder_layers[0].blocks[0].attn.kv.weight,
                                       old_kv[kv_indices][:, indices])
            output, features = target(x, return_features=True)
            self.assertEqual(output.shape, (1, 3, 3, 128, 128))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(all(f.shape[2] == width for fs in features.values() for f in fs))
            output.square().mean().backward()
            for name, parameter in target.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertEqual(len(report["mlp_hidden_indices"]), 16)
            json.dumps(report)
            source = target

    def test_calibration_selects_activation_channels_within_heads(self):
        source = ProgressiveRSTT(embed_dim=96, depths=[1] * 8, num_frames=2,
                                 head="direct", baseline_mode="separable")
        conv = source.input_proj.proj[0]
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.zero_()
            conv.weight.fill_(1)
            conv.bias.copy_(torch.arange(96))
        chosen = select_channels(source, [torch.zeros(1, 2, 3, 8, 8)], 32)
        self.assertEqual(chosen, [group * 6 + offset for group in range(16) for offset in (4, 5)])


if __name__ == "__main__":
    unittest.main()

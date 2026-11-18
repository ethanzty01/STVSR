"""CPU checks for the two-frame model's gradients and streaming time semantics."""
import sys
import unittest
import weakref
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.RSTT_tiny import RSTTTiny, interpolate_time


class TinyModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_interpolation_keeps_endpoints_and_midpoint(self):
        x = torch.tensor([2., 8.]).reshape(1, 2, 1, 1, 1)
        torch.testing.assert_close(interpolate_time(x).flatten(), torch.tensor([2., 5., 8.]))

    def test_forward_backward_and_all_parameters_participate(self):
        model = RSTTTiny(drop_path_rate=0)
        x = torch.rand(2, 2, 3, 32, 32, requires_grad=True)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 3, 3, 128, 128))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_odd_shape_and_framewise_baseline(self):
        model = RSTTTiny().eval()
        torch.nn.init.zeros_(model.conv_last.weight)
        torch.nn.init.zeros_(model.conv_last.bias)
        x = torch.rand(1, 2, 3, 17, 25)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(tuple(output.shape), (1, 3, 3, 68, 100))
        baseline = interpolate_time(x)
        expected = torch.stack([
            torch.nn.functional.interpolate(baseline[:, i], size=(68, 100),
                                            mode='bilinear', align_corners=False)
            for i in range(3)
        ], dim=1)
        torch.testing.assert_close(output, expected)

    def test_streaming_count_order_and_endpoints(self):
        model = RSTTTiny().eval()
        torch.nn.init.zeros_(model.conv_last.weight)
        torch.nn.init.zeros_(model.conv_last.bias)
        frames = (torch.full((1, 3, 32, 32), float(i)) for i in range(3))
        output = list(model.forward_sequence(frames))
        self.assertEqual(len(output), 5)
        for i, frame in enumerate(output):
            torch.testing.assert_close(frame, torch.full_like(frame, i / 2))
            self.assertFalse(frame.requires_grad)

    def test_streaming_releases_previous_output_before_next_forward(self):
        model = RSTTTiny().eval()
        previous_outputs = []

        def forward(inputs):
            self.assertTrue(all(reference() is None for reference in previous_outputs))
            output = interpolate_time(inputs).contiguous()
            previous_outputs.append(weakref.ref(output))
            return output

        model.forward = forward
        frames = (torch.full((1, 3, 4, 4), float(i)) for i in range(4))
        count = 0
        for frame in model.forward_sequence(frames):
            self.assertEqual(float(frame.mean()), count / 2)
            count += 1
            del frame
        self.assertEqual(count, 7)
        self.assertTrue(all(reference() is None for reference in previous_outputs))

    def test_streaming_empty_and_single_frame(self):
        model = RSTTTiny().eval()
        model.forward = lambda inputs: interpolate_time(inputs).contiguous()
        self.assertEqual(list(model.forward_sequence([])), [])
        frame = torch.rand(1, 3, 4, 4)
        result = list(model.forward_sequence([frame]))
        self.assertEqual(len(result), 1)
        torch.testing.assert_close(result[0], frame)


if __name__ == '__main__':
    unittest.main()

"""RSTT stages and structured weight transfer for progressive compression.

The final two-frame, 32-channel, depth-two direct-head stage has exactly the
state-dictionary layout and inference operations of :class:`RSTTTiny`.
"""

import re

import torch
from torch import nn
from torch.nn import functional as F

from .layers import DecoderLayer, Downsample, EncoderLayer, InputProj, Upsample
from .RSTT_tiny import RSTTTiny, interpolate_time


class ProgressiveRSTT(nn.Module):
    def __init__(self, embed_dim=96, depths=None, num_frames=4, head="legacy",
                 baseline_mode="original", drop_path_rate=0.1):
        super().__init__()
        depths = list(depths if depths is not None else [4] * 8)
        if embed_dim <= 0 or embed_dim % 16:
            raise ValueError("embed_dim must be a positive multiple of 16")
        if len(depths) != 8 or any(d < 1 for d in depths):
            raise ValueError("depths must contain eight positive integers")
        if num_frames not in (2, 4):
            raise ValueError("num_frames must be 2 or 4")
        if head not in ("legacy", "direct"):
            raise ValueError("head must be legacy or direct")
        if baseline_mode not in ("original", "separable"):
            raise ValueError("baseline_mode must be original or separable")
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_in_frames = num_frames
        self.num_out_frames = 2 * num_frames - 1
        self.head = head
        self.baseline_mode = baseline_mode
        self.drop_path_rate = drop_path_rate
        heads = (2, 4, 8, 16, 16, 8, 4, 2)
        self.input_proj = InputProj(3, embed_dim, kernel_size=3, stride=1)
        enc_rates = torch.linspace(0, drop_path_rate, sum(depths[:4])).tolist()
        dec_rates = torch.linspace(0, drop_path_rate, sum(depths[4:])).tolist()[::-1]
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim=embed_dim, depth=depths[i], num_heads=heads[i],
                         num_frames=num_frames, window_size=(4, 4), mlp_ratio=2.,
                         drop_path=enc_rates[sum(depths[:i]):sum(depths[:i + 1])])
            for i in range(4)
        ])
        self.downsample = nn.ModuleList([Downsample(embed_dim, embed_dim) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim=embed_dim, depth=depths[i + 4], num_heads=heads[i + 4],
                         num_frames=num_frames, window_size=(4, 4), mlp_ratio=2.,
                         drop_path=dec_rates[sum(depths[4:i + 4]):sum(depths[4:i + 5])])
            for i in range(4)
        ])
        self.upsample = nn.ModuleList([Upsample(embed_dim, embed_dim) for _ in range(3)])
        if head == "direct":
            self.conv_last = nn.Conv2d(embed_dim, 48, 3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(4)
        else:
            self.upconv1 = nn.Conv2d(embed_dim, embed_dim * 4, 3, padding=1)
            self.upconv2 = nn.Conv2d(embed_dim, 64 * 4, 3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.HRconv = nn.Conv2d(64, 64, 3, padding=1)
            self.conv_last = nn.Conv2d(64, 3, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.apply(RSTTTiny._init_weights)

    def get_config(self):
        return dict(embed_dim=self.embed_dim, depths=list(self.depths),
                    num_frames=self.num_in_frames, head=self.head,
                    baseline_mode=self.baseline_mode, drop_path_rate=self.drop_path_rate)

    @classmethod
    def from_original(cls, checkpoint_path, map_location="cpu"):
        model = cls()
        load_original_teacher(model, checkpoint_path, map_location=map_location)
        return model.to(map_location)

    _reconstruct_frame = RSTTTiny._reconstruct_frame

    def forward(self, x, return_features=False):
        if x.ndim != 5 or x.shape[1:3] != (self.num_in_frames, 3):
            raise ValueError(f"Expected [B, {self.num_in_frames}, 3, H, W] RGB input")
        height, width = x.shape[-2:]
        features = self.input_proj(x.contiguous())
        padded_h = max(32, (height + 7) // 8 * 8)
        padded_w = max(32, (width + 7) // 8 * 8)
        features = F.pad(features, (0, padded_w - width, 0, padded_h - height))
        encoded, decoded = [], []
        for i, layer in enumerate(self.encoder_layers):
            features = layer(features)
            encoded.append(features)
            if i < 3:
                features = self.downsample[i](features.contiguous())
        if self.baseline_mode == "original":
            # Match the released model's FP32 query buffer, including under AMP.
            queries = torch.empty(features.shape[0], self.num_out_frames, *features.shape[2:],
                                  device=features.device, dtype=torch.float32)
            for i in range(self.num_out_frames):
                queries[:, i] = features[:, i // 2] if i % 2 == 0 else (
                    features[:, i // 2] + features[:, i // 2 + 1]) / 2
        else:
            queries = interpolate_time(features, frames=self.num_out_frames)
        for i, layer in enumerate(self.decoder_layers):
            queries = layer(queries, encoded[3 - i])
            if return_features:
                decoded.append(queries)
            if i < 3:
                queries = self.upsample[i](queries.contiguous())
        queries = queries[..., :height, :width]
        if self.baseline_mode == "original":
            baseline = F.interpolate(x.permute(0, 2, 1, 3, 4),
                                     size=(self.num_out_frames, height * 4, width * 4),
                                     mode="trilinear", align_corners=False).permute(0, 2, 1, 3, 4)
        else:
            baseline = interpolate_time(x, frames=self.num_out_frames)
        outputs = []
        for i in range(self.num_out_frames):
            residual = self._reconstruct_frame(queries[:, i].contiguous())
            spatial_base = baseline[:, i]
            if self.baseline_mode == "separable":
                spatial_base = F.interpolate(spatial_base, size=(height * 4, width * 4),
                                             mode="bilinear", align_corners=False)
            outputs.append(residual + spatial_base)
        output = torch.stack(outputs, dim=1)
        if return_features:
            return output, {"encoder": encoded, "decoder": decoded}
        return output


def load_original_teacher(model, checkpoint, map_location="cpu"):
    """Copy a released RSTT-S checkpoint, dropping its unused fourth downsampler."""
    state = checkpoint
    if isinstance(checkpoint, (str, bytes)) or hasattr(checkpoint, "__fspath__"):
        state = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if isinstance(state, nn.Module):
        state = state.state_dict()
    for key in ("state_dict", "params", "model", "student"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    ignored = [key for key in state if key.startswith("downsample.3.")]
    state = {key: value for key, value in state.items() if key not in ignored}
    model.load_state_dict(state, strict=True)
    return {"ignored_unused_parameters": ignored}


def from_original(checkpoint_path, map_location="cpu"):
    return ProgressiveRSTT.from_original(checkpoint_path, map_location=map_location)


def _normalise_score(score):
    return score / score.mean().clamp_min(1e-12)


@torch.no_grad()
def select_channels(source, calibration_inputs, target_dim, max_batches=32):
    """Rank channels, keeping equal counts in each of the 16 finest attention heads.

    Input-projection and all encoder/decoder activation RMS scores are normalized
    per layer and averaged, then combined with projection-weight RMS.
    Residual pathways use one shared channel map. Subsequent pruning operates on
    the current model, so selections stay nested throughout the width schedule.
    Pass ``None`` for calibration_inputs to use weight importance alone.
    """
    width = source.embed_dim
    if target_dim <= 0 or target_dim > width or target_dim % 16:
        raise ValueError("target_dim must be a positive multiple of 16 up to source width")
    weight = source.input_proj.proj[0].weight.detach().float()
    score = _normalise_score(weight.square().mean((1, 2, 3)).sqrt())
    if calibration_inputs is not None:
        sums = [torch.zeros(width, device=weight.device) for _ in range(9)]
        counts = [0] * 9
        parameter = next(source.parameters())
        projected = []
        handle = source.input_proj.register_forward_hook(lambda module, inputs, output: projected.append(output))
        was_training = source.training
        source.eval()
        try:
            for batch_idx, batch in enumerate(calibration_inputs):
                if batch_idx >= max_batches:
                    break
                if isinstance(batch, dict):
                    key = "teacher_lr" if source.num_in_frames == 4 else "student_lr"
                    batch = batch[key]
                elif isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(device=parameter.device, dtype=parameter.dtype)
                projected.clear()
                _, features = source(batch, return_features=True)
                activations = projected + features["encoder"] + features["decoder"]
                for i, activation in enumerate(activations):
                    activation = activation.float()
                    sums[i] += activation.square().sum((0, 1, 3, 4))
                    counts[i] += activation.numel() // width
            if counts[0]:
                scores = [_normalise_score((total / count).sqrt())
                          for total, count in zip(sums, counts)]
                score += torch.stack(scores).mean(0)
        finally:
            handle.remove()
            source.train(was_training)
    group_size, keep_per_group = width // 16, target_dim // 16
    chosen = []
    for group in range(16):
        start = group * group_size
        local = score[start:start + group_size].argsort(descending=True, stable=True)[:keep_per_group]
        chosen.extend((local.sort().values + start).tolist())
    return chosen


def _source_key(key, block_pairs):
    match = re.match(r"((?:encoder|decoder)_layers\.\d+)\.blocks\.(\d+)(\..*)", key)
    if match and match[1] in block_pairs:
        return f"{match[1]}.blocks.{block_pairs[match[1]][int(match[2])]}{match[3]}"
    return key


def _validate_channels(indices, old_width, new_width):
    if len(indices) != new_width or len(set(indices)) != new_width:
        raise ValueError("channel_indices must contain target-width unique source channels")
    if indices != sorted(indices) or min(indices) < 0 or max(indices) >= old_width:
        raise ValueError("channel_indices must be sorted and within the source width")
    counts = [sum(i // (old_width // 16) == group for i in indices) for group in range(16)]
    if counts != [new_width // 16] * 16:
        raise ValueError("Select equal channel counts in each of the 16 attention-head groups")


@torch.no_grad()
def migrate_model(source, target_config, channel_indices=None, block_pairs=None):
    """Transfer a stage through head replacement, temporal/depth reduction, or pruning.

    ``block_pairs`` maps stage paths, e.g. ``encoder_layers.0``, to source block
    IDs. ``[0, 1]`` and ``[2, 3]`` both preserve regular/shifted-window order.
    Relative-position indices are regenerated; bias tables retain matching
    signed temporal offsets. Every MLP hidden layer gets an independent ranking.
    """
    config = source.get_config()
    config.update(target_config)
    target = ProgressiveRSTT(**config)
    parameter = next(source.parameters())
    target.to(device=parameter.device, dtype=parameter.dtype)
    old_width, new_width = source.embed_dim, target.embed_dim
    if new_width > old_width:
        raise ValueError("Progressive migration supports unchanged or reduced widths")
    if source.head == "direct" and target.head == "legacy":
        raise ValueError("Head migration follows legacy -> direct")
    block_pairs = dict(block_pairs or {})
    for side in ("encoder", "decoder"):
        old_layers = getattr(source, f"{side}_layers")
        new_layers = getattr(target, f"{side}_layers")
        for layer_idx, (old_layer, new_layer) in enumerate(zip(old_layers, new_layers)):
            path = f"{side}_layers.{layer_idx}"
            mapping = list(block_pairs.get(path, range(len(new_layer.blocks))))
            if len(mapping) != len(new_layer.blocks) or len(set(mapping)) != len(mapping):
                raise ValueError(f"Block mapping length/uniqueness mismatch for {path}")
            if any(i < 0 or i >= len(old_layer.blocks) or i % 2 != j % 2
                   for j, i in enumerate(mapping)):
                raise ValueError(f"Block mapping must preserve regular/shifted parity for {path}")
            block_pairs[path] = mapping
    selection_method = "provided" if channel_indices is not None else "weight_rms"
    if channel_indices is None:
        channel_indices = select_channels(source, None, new_width)
    channel_indices = list(channel_indices)
    _validate_channels(channel_indices, old_width, new_width)
    channels = torch.tensor(channel_indices, device=parameter.device, dtype=torch.long)
    kv_channels = torch.cat((channels, channels + old_width))
    shuffled_channels = (channels[:, None] * 4 + torch.arange(4, device=channels.device)).flatten()
    old_state, new_state = source.state_dict(), target.state_dict()
    hidden_indices = {}
    report = {"source_config": source.get_config(), "target_config": target.get_config(),
              "channel_indices": channel_indices, "channel_selection": selection_method,
              "block_pairs": block_pairs, "mlp_hidden_indices": {},
              "temporal_bias_crops": {}, "fresh_parameters": [], "copied_tensors": 0}
    for key in new_state:
        if not key.endswith("mlp.fc1.weight"):
            continue
        prefix = key[:-len("fc1.weight")]
        old_prefix = _source_key(prefix, block_pairs)
        fc1 = old_state[old_prefix + "fc1.weight"].float()
        fc2 = old_state[old_prefix + "fc2.weight"].float()
        if new_width == old_width:
            selected = torch.arange(2 * old_width, device=parameter.device)
        else:
            importance = _normalise_score(fc1.square().mean(1).sqrt())
            importance += _normalise_score(fc2.square().mean(0).sqrt())
            selected = importance.argsort(descending=True, stable=True)[:2 * new_width].sort().values
        hidden_indices[prefix] = selected
        report["mlp_hidden_indices"][prefix] = selected.tolist()

    for key, value in new_state.items():
        if key.endswith("relative_position_index"):
            continue
        old_key = _source_key(key, block_pairs)
        if source.head != target.head and key.startswith("conv_last."):
            value.zero_()
            report["fresh_parameters"].append(key)
            continue
        if old_key not in old_state:
            raise ValueError(f"Missing source tensor for {key}: {old_key}")
        old = old_state[old_key]
        if key.endswith("relative_position_bias_table"):
            if old.shape[0] != value.shape[0]:
                old_times, new_times = old.shape[0] // 49, value.shape[0] // 49
                if new_times > old_times or (old_times - new_times) % 2:
                    raise ValueError("Temporal bias migration requires a centered offset subset")
                start = (old_times - new_times) // 2
                old = old.reshape(old_times, 49, old.shape[1])[start:start + new_times].reshape_as(value)
                report["temporal_bias_crops"][key] = {"source_offset_start": start,
                                                       "source_temporal_bins": old_times,
                                                       "target_temporal_bins": new_times}
        elif old_width != new_width:
            if key.startswith("input_proj."):
                old = old.index_select(0, channels)
            elif ".mlp." in key:
                prefix, suffix = key.rsplit("mlp.", 1)
                hidden = hidden_indices[prefix + "mlp."]
                if suffix == "fc1.weight":
                    old = old.index_select(0, hidden).index_select(1, channels)
                elif suffix == "fc1.bias":
                    old = old.index_select(0, hidden)
                elif suffix == "fc2.weight":
                    old = old.index_select(0, channels).index_select(1, hidden)
                else:
                    old = old.index_select(0, channels)
            elif key.startswith(("encoder_layers.", "decoder_layers.")):
                rows = kv_channels if ".kv." in key else channels
                old = old.index_select(0, rows)
                if old.ndim == 2:
                    old = old.index_select(1, channels)
            elif key.startswith(("downsample.", "upsample.")):
                old = old.index_select(0, channels)
                if old.ndim == 4:
                    old = old.index_select(1, channels)
            elif key.startswith("conv_last.") and source.head == "direct":
                if old.ndim == 4:
                    old = old.index_select(1, channels)
            elif key.startswith("upconv1."):
                old = old.index_select(0, shuffled_channels)
                if old.ndim == 4:
                    old = old.index_select(1, channels)
            elif key == "upconv2.weight":
                old = old.index_select(1, channels)
        if old.shape != value.shape:
            raise ValueError(f"Transfer shape mismatch at {key}: {tuple(old.shape)} -> {tuple(value.shape)}")
        value.copy_(old)
        report["copied_tensors"] += 1
    target.load_state_dict(new_state, strict=True)
    target.train(source.training)
    return target, report

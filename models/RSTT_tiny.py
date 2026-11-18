"""Two-frame RSTT student with separable interpolation and framewise reconstruction."""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import DecoderLayer, Downsample, EncoderLayer, InputProj, Upsample


def interpolate_time(x, frames=3):
    """Interpolate BTCHW features at endpoint-aligned, evenly spaced times."""
    batch, time, channels, height, width = x.shape
    x = x.permute(0, 2, 3, 4, 1).reshape(batch, channels * height * width, time)
    x = F.interpolate(x, size=frames, mode="linear", align_corners=True)
    return x.reshape(batch, channels, height, width, frames).permute(0, 4, 1, 2, 3)


class RSTTTiny(nn.Module):
    """Map [B, 2, 3, H, W] to [B, 3, 3, 4H, 4W].

    ``depth`` counts individual attention blocks. A depth of two restores the
    alternating regular/shifted-window pair at every encoder and decoder stage.
    The direct head projects to 48 channels then applies PixelShuffle(4).
    ``head='legacy'`` loads the first experiment's reconstruction weights.
    """

    def __init__(self, embed_dim=32, depth=2, drop_path_rate=0.1, head='direct'):
        super().__init__()
        heads = (2, 4, 8, 16, 16, 8, 4, 2)
        if embed_dim <= 0 or any(embed_dim % head for head in heads) or depth < 1:
            raise ValueError("Use a positive channel multiple of 16 and depth >= 1.")
        self.embed_dim = embed_dim
        self.depth = depth
        if head not in ('direct', 'legacy'):
            raise ValueError('head must be direct or legacy')
        self.head = head
        self.drop_path_rate = drop_path_rate
        self.num_in_frames = 2
        self.num_out_frames = 3
        self.input_proj = InputProj(3, embed_dim, kernel_size=3, stride=1)
        rates = torch.linspace(0, drop_path_rate, 4 * depth).tolist()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim=embed_dim, depth=depth, num_heads=heads[i],
                         num_frames=2, window_size=(4, 4), mlp_ratio=2.,
                         drop_path=rates[i * depth:(i + 1) * depth])
            for i in range(4)
        ])
        # Exactly three downsampling operations are executed in the four-level U.
        self.downsample = nn.ModuleList([Downsample(embed_dim, embed_dim) for _ in range(3)])
        rates.reverse()
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim=embed_dim, depth=depth, num_heads=heads[i + 4],
                         num_frames=2, window_size=(4, 4), mlp_ratio=2.,
                         drop_path=rates[i * depth:(i + 1) * depth])
            for i in range(4)
        ])
        self.upsample = nn.ModuleList([Upsample(embed_dim, embed_dim) for _ in range(3)])
        if head == 'direct':
            self.conv_last = nn.Conv2d(embed_dim, 4 * 4 * 3, 3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(4)
        else:
            self.upconv1 = nn.Conv2d(embed_dim, embed_dim * 4, 3, padding=1)
            self.upconv2 = nn.Conv2d(embed_dim, 64 * 4, 3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.HRconv = nn.Conv2d(64, 64, 3, padding=1)
            self.conv_last = nn.Conv2d(64, 3, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _reconstruct_frame(self, feature):
        if self.head == 'direct':
            return self.pixel_shuffle(self.conv_last(feature))
        feature = self.lrelu(self.pixel_shuffle(self.upconv1(feature)))
        feature = self.lrelu(self.pixel_shuffle(self.upconv2(feature)))
        return self.conv_last(self.lrelu(self.HRconv(feature)))

    def forward(self, x):
        if x.ndim != 5 or x.shape[1:3] != (2, 3):
            raise ValueError("Expected [B, 2, 3, H, W] RGB input.")
        height, width = x.shape[-2:]
        features = self.input_proj(x.contiguous())
        # The reused attention uses a 4x4 relative-position table at every scale.
        padded_h = max(32, (height + 7) // 8 * 8)
        padded_w = max(32, (width + 7) // 8 * 8)
        features = F.pad(features, (0, padded_w - width, 0, padded_h - height))
        encoded = []
        for i, layer in enumerate(self.encoder_layers):
            features = layer(features)
            encoded.append(features)
            if i < 3:
                features = self.downsample[i](features.contiguous())
        queries = interpolate_time(features)
        for i, layer in enumerate(self.decoder_layers):
            queries = layer(queries, encoded[3 - i])
            if i < 3:
                queries = self.upsample[i](queries.contiguous())
        queries = queries[..., :height, :width]
        # Build the low-resolution temporal baseline only when it is needed.
        baseline = interpolate_time(x)
        outputs = []
        for i in range(3):
            residual = self._reconstruct_frame(queries[:, i].contiguous())
            spatial_base = F.interpolate(baseline[:, i], size=(height * 4, width * 4),
                                         mode="bilinear", align_corners=False)
            outputs.append(residual + spatial_base)
        return torch.stack(outputs, dim=1)

    def forward_sequence(self, frames):
        """Yield 2N-1 HR frames from an iterable of [B, 3, H, W] LR frames.

        Call ``model.eval()`` before use. Each pair emits its first endpoint and
        midpoint; the last endpoint is emitted after the input iterator ends.
        Write or consume yielded frames as they arrive to keep buffering bounded.
        Release GPU frame views after consumption, before requesting the next frame.
        """
        if self.training:
            raise ValueError("Call eval() before streaming inference.")
        frames = iter(frames)
        previous = next(frames, None)
        if previous is None:
            return
        current = next(frames, None)
        if current is None:
            with torch.no_grad():
                output = self(torch.stack((previous, previous), dim=1))
            yield output[:, 0]
            return
        while True:
            with torch.no_grad():
                output = self(torch.stack((previous, current), dim=1))
            yield output[:, 0]
            yield output[:, 1]
            previous = current
            current = next(frames, None)
            if current is None:
                yield output[:, 2]
                return
            # A final-frame view would keep the entire three-frame output alive.
            del output

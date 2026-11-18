"""Aligned video samples for four-frame teacher / two-frame student training."""
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class VimeoDistillationDataset(Dataset):
    """Use a held-out part of train for validation and official test records.

    A student sees adjacent entries j,j+1 of the teacher's four LR frames.
    Its three HR targets are frames 2*j through 2*j+2 of the seven-frame clip.
    Training applies shared transforms; validation/test use the middle triple.
    ``patch_size=None`` preserves the full frame; evaluation patches are centered.
    """

    def __init__(self, root, split='train', patch_size=128,
                 validation_count=128, seed=2026, limit=None):
        self.root = Path(root)
        self.split = split
        self.training = split == 'train'
        self.patch_size = patch_size
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test')
        if patch_size is not None and (patch_size < 16 or patch_size % 4):
            raise ValueError('patch_size must be a multiple of four, at least 16')
        source_split = 'test' if split == 'test' else 'train'
        manifest = self.root / source_split / 'Vimeo-90k' / 'manifest.json'
        records = json.loads(manifest.read_text())['records']
        records.sort(key=lambda item: item['id'])
        if split == 'test':
            records = [r for r in records if r.get('evaluation_included', True)]
        else:
            if not 0 < validation_count < len(records):
                raise ValueError('validation_count must leave nonempty train/val sets')
            random.Random(seed).shuffle(records)
            records = records[validation_count:] if self.training else records[:validation_count]
        if limit is not None:
            if limit < 1:
                raise ValueError('limit must be positive')
            # A fixed, seeded sample avoids selecting only early directory IDs.
            random.Random(seed + 1).shuffle(records)
            records = records[:limit]
        self.records = records
        if not self.records:
            raise ValueError('No clips selected')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        hr = torch.load(self.root / record['file'], map_location='cpu',
                        weights_only=True)['hr']
        if hr.dtype != torch.uint8 or hr.ndim != 4 or hr.shape[:2] != (7, 3):
            raise ValueError(f"Expected uint8 [7,3,H,W]: {record['id']}")
        hr = hr.float().div_(255)
        height, width = hr.shape[-2:]
        hr = hr[..., :height - height % 4, :width - width % 4]
        if self.training and random.random() < 0.5:
            hr = hr.flip(0)
        teacher_lr = F.interpolate(hr[::2], scale_factor=0.25, mode='bicubic',
                                   align_corners=False, antialias=True).clamp_(0, 1)
        if self.patch_size is not None:
            p = self.patch_size // 4
            h, w = teacher_lr.shape[-2:]
            if p > min(h, w):
                raise ValueError(f"Patch exceeds frame size: {record['id']}")
            if self.training:
                top, left = random.randrange(h - p + 1), random.randrange(w - p + 1)
            else:
                top, left = (h - p) // 2, (w - p) // 2
            teacher_lr = teacher_lr[..., top:top + p, left:left + p]
            hr = hr[..., 4 * top:4 * (top + p), 4 * left:4 * (left + p)]
        if self.training:
            for dim in (-1, -2):
                if random.random() < 0.5:
                    hr, teacher_lr = hr.flip(dim), teacher_lr.flip(dim)
            if random.random() < 0.5:
                hr, teacher_lr = hr.transpose(-1, -2), teacher_lr.transpose(-1, -2)
        pair = random.randrange(3) if self.training else 1
        return {
            'teacher_lr': teacher_lr.contiguous(),
            'student_lr': teacher_lr[pair:pair + 2].contiguous(),
            'gt': hr[2 * pair:2 * pair + 3].contiguous(),
            'pair_index': pair,
            'key': record['id'],
            'group': record.get('motion_group', self.split),
        }


class REDSDistillationDataset(Dataset):
    """Seven consecutive REDS frames with the same time alignment as Vimeo.

    Each manifest sequence contributes ``len(sequence) - 6`` windows. Validation
    and test use each window's middle triple; their frame observations overlap.
    Patch loading reads an aligned uint8 ROI plus an eight-pixel bicubic halo
    before converting to float. This preserves full-frame x4 antialiased
    downsampling at the selected patch, including image-edge behavior.
    """

    def __init__(self, root, split='train', patch_size=128, seed=2026, limit=None):
        self.root = Path(root)
        self.split = split
        self.training = split == 'train'
        self.patch_size = patch_size
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test')
        if patch_size is not None and (patch_size < 16 or patch_size % 4):
            raise ValueError('patch_size must be a multiple of four, at least 16')
        manifest = json.loads((self.root / split / 'REDS' / 'manifest.json').read_text())
        records = []
        for sequence, paths in sorted(manifest['sequences'].items()):
            paths = sorted(paths)
            numbers = [int(Path(path).stem) for path in paths]
            if any(b != a + 1 for a, b in zip(numbers, numbers[1:])):
                raise ValueError(f'Nonconsecutive REDS sequence: {sequence}')
            for start in range(len(paths) - 6):
                records.append({'id': f'REDS/{sequence}/{numbers[start]:08d}',
                                'sequence': sequence, 'start': start,
                                'files': paths[start:start + 7]})
        if limit is not None:
            if limit < 1:
                raise ValueError('limit must be positive')
            random.Random(seed + 1).shuffle(records)
            records = records[:limit]
        self.records = records
        if not records:
            raise ValueError('No REDS clips selected')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        frames = [torch.load(self.root / path, map_location='cpu',
                             weights_only=True, mmap=True)['hr']
                  for path in record['files']]
        shape = frames[0].shape
        if any(frame.dtype != torch.uint8 or frame.ndim != 4 or
               frame.shape[:2] != (1, 3) or frame.shape != shape for frame in frames):
            raise ValueError(f"Expected matching uint8 [1,3,H,W] frames: {record['id']}")
        height, width = (value - value % 4 for value in shape[-2:])
        top, left, bottom, right = 0, 0, height, width
        if self.patch_size is not None:
            p = self.patch_size // 4
            h, w = height // 4, width // 4
            if p > min(h, w):
                raise ValueError(f"Patch exceeds frame size: {record['id']}")
            if self.training:
                crop_top, crop_left = random.randrange(h - p + 1), random.randrange(w - p + 1)
            else:
                crop_top, crop_left = (h - p) // 2, (w - p) // 2
            top, left = max(0, 4 * crop_top - 8), max(0, 4 * crop_left - 8)
            bottom = min(height, 4 * (crop_top + p) + 8)
            right = min(width, 4 * (crop_left + p) + 8)
        hr = torch.cat([frame[..., top:bottom, left:right] for frame in frames]).float().div_(255)
        if self.training and random.random() < 0.5:
            hr = hr.flip(0)
        teacher_lr = F.interpolate(hr[::2], scale_factor=0.25, mode='bicubic',
                                   align_corners=False, antialias=True).clamp_(0, 1)
        if self.patch_size is not None:
            local_top, local_left = crop_top - top // 4, crop_left - left // 4
            teacher_lr = teacher_lr[..., local_top:local_top + p, local_left:local_left + p]
            hr = hr[..., 4 * local_top:4 * (local_top + p), 4 * local_left:4 * (local_left + p)]
        if self.training:
            for dim in (-1, -2):
                if random.random() < 0.5:
                    hr, teacher_lr = hr.flip(dim), teacher_lr.flip(dim)
            if random.random() < 0.5:
                hr, teacher_lr = hr.transpose(-1, -2), teacher_lr.transpose(-1, -2)
        pair = random.randrange(3) if self.training else 1
        return {'teacher_lr': teacher_lr.contiguous(),
                'student_lr': teacher_lr[pair:pair + 2].contiguous(),
                'gt': hr[2 * pair:2 * pair + 3].contiguous(),
                'pair_index': pair, 'key': record['id'], 'group': 'REDS'}

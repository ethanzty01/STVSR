"""Check teacher/student time alignment and held-out split behavior."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill_data import REDSDistillationDataset, VimeoDistillationDataset


class DistillationDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for split in ('train', 'test'):
            directory = self.root / split / 'Vimeo-90k'
            directory.mkdir(parents=True)
            records = []
            for i in range(6):
                hr = (torch.arange(7, dtype=torch.uint8) * 30).view(7, 1, 1, 1)
                hr = hr.expand(7, 3, 32, 48).contiguous()
                path = directory / f'{i:03d}.pt'
                torch.save({'hr': hr}, path)
                records.append({'id': f'{i:03d}', 'file': str(path.relative_to(self.root)),
                                'evaluation_included': i != 5, 'motion_group': 'slow'})
            (directory / 'manifest.json').write_text(json.dumps({'records': records}))

    def tearDown(self):
        self.temp.cleanup()

    def test_partitions_and_test_filter(self):
        train = VimeoDistillationDataset(self.root, 'train', validation_count=2)
        val = VimeoDistillationDataset(self.root, 'val', validation_count=2)
        train_ids = {r['id'] for r in train.records}
        val_ids = {r['id'] for r in val.records}
        self.assertEqual(len(train_ids | val_ids), 6)
        self.assertFalse(train_ids & val_ids)
        repeated = VimeoDistillationDataset(self.root, 'val', validation_count=2)
        self.assertEqual(val.records, repeated.records)
        test = VimeoDistillationDataset(self.root, 'test')
        self.assertEqual(len(test), 5)
        self.assertNotIn('005', {r['id'] for r in test.records})

    def test_temporal_alignment_after_augmentation(self):
        ds = VimeoDistillationDataset(self.root, 'train', patch_size=16, validation_count=2)
        for seed in range(6):
            random.seed(seed)
            item = ds[0]
            j = item['pair_index']
            self.assertEqual(tuple(item['student_lr'].shape), (2, 3, 4, 4))
            self.assertEqual(tuple(item['gt'].shape), (3, 3, 16, 16))
            torch.testing.assert_close(item['student_lr'], item['teacher_lr'][j:j+2])
            torch.testing.assert_close(item['student_lr'][:, :, 0, 0],
                                       item['gt'][::2, :, 0, 0], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(item['gt'][1], (item['gt'][0] + item['gt'][2]) / 2)

    def test_full_frame_evaluation_middle_triple(self):
        ds = VimeoDistillationDataset(self.root, 'test', patch_size=None)
        item = ds[0]
        self.assertEqual(item['pair_index'], 1)
        self.assertEqual(tuple(item['teacher_lr'].shape), (4, 3, 8, 12))
        self.assertEqual(tuple(item['gt'].shape), (3, 3, 32, 48))
        torch.testing.assert_close(item['gt'][:, 0, 0, 0], torch.tensor([60., 90., 120.]) / 255)


class REDSDistillationDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        generator = torch.Generator().manual_seed(42)
        self.hr = torch.randint(0, 256, (9, 3, 64, 80), generator=generator, dtype=torch.uint8)
        for split, sequences in [('train', ('001', '002')), ('val', ('240',)), ('test', ('000',))]:
            paths = {}
            for sequence in sequences:
                directory = self.root / split / 'REDS' / sequence
                directory.mkdir(parents=True)
                paths[sequence] = []
                for frame in range(len(self.hr)):
                    path = directory / f'{frame:08d}.pt'
                    torch.save({'hr': self.hr[frame:frame + 1]}, path)
                    paths[sequence].append(str(path.relative_to(self.root)))
            (self.root / split / 'REDS' / 'manifest.json').write_text(json.dumps({'sequences': paths}))

    def tearDown(self):
        self.temp.cleanup()

    def test_windows_stay_in_sequence_and_split(self):
        datasets = {split: REDSDistillationDataset(self.root, split) for split in ('train', 'val', 'test')}
        self.assertEqual([len(datasets[s]) for s in ('train', 'val', 'test')], [6, 3, 3])
        ids = [{r['id'] for r in ds.records} for ds in datasets.values()]
        self.assertEqual(sum(map(len, ids)), len(set.union(*ids)))
        for split, ds in datasets.items():
            for record in ds.records:
                self.assertTrue(all(path.startswith(f'{split}/REDS/{record["sequence"]}/')
                                    for path in record['files']))
                numbers = [int(Path(path).stem) for path in record['files']]
                self.assertEqual(numbers, list(range(record['start'], record['start'] + 7)))

    def test_roi_matches_full_frame_bicubic_at_edges_and_interior(self):
        dataset = REDSDistillationDataset(self.root, 'train', patch_size=16)
        full = self.hr[:7].float() / 255
        lr = F.interpolate(full[::2], scale_factor=.25, mode='bicubic',
                           align_corners=False, antialias=True).clamp(0, 1)
        # Exercise both image boundaries and patches with a complete filter halo.
        for top, left in [(0, 0), (12, 16), (3, 7), (0, 9), (8, 0)]:
            with patch('distill_data.random.randrange', side_effect=[top, left, 1]), \
                 patch('distill_data.random.random', return_value=1.):
                item = dataset[0]
            torch.testing.assert_close(item['teacher_lr'], lr[..., top:top + 4, left:left + 4],
                                       atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(item['gt'], full[2:5, :, 4 * top:4 * top + 16, 4 * left:4 * left + 16])
            torch.testing.assert_close(item['student_lr'], item['teacher_lr'][1:3])

    def test_full_frame_test_middle_triple(self):
        dataset = REDSDistillationDataset(self.root, 'test', patch_size=None)
        item = dataset[2]
        self.assertEqual(item['key'], 'REDS/000/00000002')
        self.assertEqual(tuple(item['teacher_lr'].shape), (4, 3, 16, 20))
        torch.testing.assert_close(item['gt'], self.hr[4:7].float() / 255)

    def test_missing_temporal_frame_is_detected(self):
        path = self.root / 'test' / 'REDS' / 'manifest.json'
        manifest = json.loads(path.read_text())
        del manifest['sequences']['000'][4]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Nonconsecutive'):
            REDSDistillationDataset(self.root, 'test')


if __name__ == '__main__':
    unittest.main()

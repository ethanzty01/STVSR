"""Check temporal endpoint choice and unique sequence frame identities."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_progressive import score, stitch_pairs, summarize, unique_vid4_windows


def test_three_pairs_keep_first_occurrence_of_shared_endpoints():
    # Distinct predictions for the same endpoint expose accidental replacement.
    pairs = [torch.tensor(values).reshape(1, 3, 1, 1, 1)
             for values in ([0, 1, 2], [20, 3, 4], [40, 5, 6])]
    assert stitch_pairs(pairs).flatten().tolist() == list(range(7))


def test_vid4_frame_labels_cover_each_sequence_exactly_once():
    for length in (7, 8, 13, 34, 41, 47, 49):
        labels = [indices[local] for indices, keep in unique_vid4_windows(length) for local in keep]
        assert labels == list(range(length))
    assert sum(len(keep) for _, keep in unique_vid4_windows(34, limit=1)) == 7


def test_vid4_score_uses_global_indices_and_actual_input_timestamp_kind():
    target = torch.zeros(1, 7, 3, 11, 11)
    prediction = torch.ones_like(target) * .1
    # City has an odd-start trailing window. Global frame 31 is an observed endpoint.
    rows = score({'progressive': prediction}, target, ['city'], ['city'],
                 [list(range(27, 34))], keep=[4, 5, 6])
    assert [r['frame'] for r in rows] == [31, 32, 33]
    assert [r['kind'] for r in rows] == ['endpoints', 'center', 'endpoints']
    assert all(r['window_start'] == 27 for r in rows)


def test_vid4_summary_weights_sequences_equally_despite_different_lengths():
    rows = [{'model': 'progressive', 'group': group, 'kind': kind,
             'psnr_y': value, 'psnr_rgb': value, 'ssim_y': value / 100}
            for group, count, value in [('short', 2, 20.), ('long', 4, 30.)]
            for kind in ('endpoints', 'center') for _ in range(count)]
    summary = summarize(rows, equal_sequence=True)['progressive']
    assert summary['equal_sequence_mean']['all']['psnr_y'] == 25.
    assert abs(summary['all']['all']['psnr_y'] - 80 / 3) < 1e-10

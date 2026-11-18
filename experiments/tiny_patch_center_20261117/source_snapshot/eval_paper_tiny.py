"""Seven-frame Vimeo and four-sequence Vid4 comparison for tiny and RSTT-S."""
import csv
import json
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from distill_data import VimeoDistillationDataset
from distill_metrics import compute_metrics
from models.RSTT import RSTT
from models.RSTT_tiny import RSTTTiny
from utils.eval_utils import index_generation

ROOT = Path('../Datasets_pt')
OUT = Path('experiments/tiny_mixed_200k_b16/paper_comparison')


class VimeoSeven(VimeoDistillationDataset):
    def __getitem__(self, index):
        record = self.records[index]
        hr = torch.load(self.root / record['file'], map_location='cpu', weights_only=True)['hr'].float().div_(255)
        lr = F.interpolate(hr[::2], scale_factor=.25, mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1)
        # Official evaluation reads 8-bit LR PNGs.
        lr = lr.mul(255).round().div(255)
        return {'lr': lr, 'hr': hr, 'key': record['id'], 'group': record['motion_group']}


def means(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in ('psnr_rgb', 'psnr_y', 'ssim_y')}


@torch.inference_mode()
def predict(models, lr):
    result = {'teacher': models['teacher'](lr).cpu()}
    for name in ('tiny_best', 'tiny_last'):
        pair = [models[name](lr[:, i:i+2]) for i in range(3)]
        result[name] = torch.cat([pair[0], pair[1][:, 1:], pair[2][:, 1:]], dim=1).cpu()
    return result


def score(outputs, hr, keys, groups, indices=None):
    rows = []
    for name, pred in outputs.items():
        frames = compute_metrics(pred, hr)['per_frame']
        for b, clip in enumerate(frames):
            for i, metrics in enumerate(clip):
                if indices is not None and i not in indices:
                    continue
                rows.append({'model': name, 'key': keys[b], 'group': groups[b], 'frame': i, **metrics})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    models = {}
    for name, filename in [('tiny_best', 'distilled_best.pth'), ('tiny_last', 'distilled_last.pth')]:
        ckpt = torch.load(OUT.parent / filename, map_location='cpu', weights_only=True)
        model = RSTTTiny(**ckpt['model_config']).to(device).eval()
        model.load_state_dict(ckpt['state_dict'])
        models[name] = model
    teacher = RSTT(depths=[4]*8).to(device).eval()
    teacher.load_state_dict(torch.load('checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth', map_location=device, weights_only=True))
    models['teacher'] = teacher
    vimeo = VimeoSeven(ROOT, 'test', patch_size=None)
    rows = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = deque()
        for i, batch in enumerate(DataLoader(vimeo, batch_size=4, num_workers=2)):
            outputs = predict(models, batch['lr'].to(device))
            pending.append(pool.submit(score, outputs, batch['hr'], batch['key'], batch['group']))
            if len(pending) >= 4:
                rows.extend(pending.popleft().result())
            if i % 100 == 0:
                print(json.dumps({'dataset':'vimeo','clips':min((i+1)*4,len(vimeo)),'total':len(vimeo),'seconds':time.perf_counter()-started}), flush=True)
        for f in pending:
            rows.extend(f.result())
    with (OUT/'vimeo_frames.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    result = {'protocol': 'Seven unique frames per Vimeo clip; Vid4 each frame scored once and sequences equally weighted; Y-channel on rounded RGB8, no crop; PyTorch antialiased bicubic x4 then rounded LR8; first occurrence of duplicate endpoint; FP32. Official code uses saved bicubic PNG; generating LR with PyTorch can differ from official Matlab resize.', 'vimeo_clips':len(vimeo), 'vimeo':{}, 'vid4':{}}
    for model in models:
        result['vimeo'][model] = {group: means([r for r in rows if r['model']==model and r['group']==group]) for group in ('fast','medium','slow')}
    print(json.dumps({'vimeo_results':result['vimeo']}),flush=True)
    manifest = json.loads((ROOT/'test/Vid4/manifest.json').read_text())
    vidrows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=deque()
        for sequence, paths in sorted(manifest['sequences'].items()):
            frames=torch.cat([torch.load(ROOT/p, map_location='cpu', weights_only=True)['hr'] for p in sorted(paths)]).float().div_(255)
            h,w=frames.shape[-2:]; frames=frames[...,:h-h%4,:w-w%4]
            lr=F.interpolate(frames, scale_factor=.25, mode='bicubic', align_corners=False, antialias=True).clamp_(0,1).mul_(255).round_().div_(255)
            seen=set()
            for indices in index_generation(7,len(frames)):
                unique=[i for i,idx in enumerate(indices) if idx not in seen]
                seen.update(indices)
                outputs=predict(models,lr[indices[::2]][None].to(device))
                pending.append(pool.submit(score,outputs,frames[indices][None],[sequence],[sequence],unique))
                if len(pending)>=4: vidrows.extend(pending.popleft().result())
            print(json.dumps({'dataset':'Vid4','sequence':sequence,'frames':len(frames)}),flush=True)
        for f in pending: vidrows.extend(f.result())
    with (OUT/'vid4_frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=vidrows[0].keys());writer.writeheader();writer.writerows(vidrows)
    for model in models:
        per_sequence={seq:means([r for r in vidrows if r['model']==model and r['group']==seq]) for seq in manifest['sequences']}
        result['vid4'][model]={'sequences':per_sequence,'equal_sequence_mean':{k:float(np.mean([v[k] for v in per_sequence.values()])) for k in ('psnr_rgb','psnr_y','ssim_y')}}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'vid4_results':result['vid4'],'path':str(OUT/'results.json')}),flush=True)


if __name__ == '__main__':
    main()

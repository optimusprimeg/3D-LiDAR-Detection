#!/usr/bin/env python
"""Batch inference on multiple A9 KITTI frames."""

from pathlib import Path
import numpy as np
import torch
from pointpillars.model import PointPillars
from pointpillars.utils import read_points


def main():
    ckpt = 'pretrained/epoch_160.pth'
    velodyne_dir = Path('../kitti_format/training/velodyne')
    files = sorted(velodyne_dir.glob('*.bin'))[:20]  # First 20 frames
    score_thr = 0.005
    point_range = np.array([-50, -50, -8, 49.84, 49.84, 5], dtype=np.float32)
    class_map = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
    out_file = Path('batch_inference_summary.txt')

    model = PointPillars(nclasses=3)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.score_thr = score_thr
    model.eval()

    lines = []
    lines.append(f'A9 BATCH INFERENCE RESULTS')
    lines.append(f'==========================')
    lines.append(f'checkpoint: {ckpt}')
    lines.append(f'frames: {len(files)}')
    lines.append(f'score_thr: {score_thr}')
    lines.append(f'')
    lines.append('frame | boxes | top_score | class_breakdown')
    lines.append('----- | ----- | --------- | ---------------')

    total_boxes = 0
    total_ped = 0
    total_cyc = 0
    total_car = 0

    for f in files:
        pc = read_points(str(f))
        m = (
            (pc[:, 0] > point_range[0]) & (pc[:, 0] < point_range[3]) &
            (pc[:, 1] > point_range[1]) & (pc[:, 1] < point_range[4]) &
            (pc[:, 2] > point_range[2]) & (pc[:, 2] < point_range[5])
        )
        pc = pc[m]

        with torch.no_grad():
            out = model(batched_pts=[torch.from_numpy(pc).float()], mode='test')[0]

        if isinstance(out, dict):
            labels = np.asarray(out.get('labels', []))
            scores = np.asarray(out.get('scores', []))
            n_boxes = len(labels)
            top_score = float(scores.max()) if len(scores) else 0.0
            
            # Count by class
            ped = int((labels == 0).sum())
            cyc = int((labels == 1).sum())
            car = int((labels == 2).sum())
            total_boxes += n_boxes
            total_ped += ped
            total_cyc += cyc
            total_car += car
            
            breakdown = f'Ped:{ped}, Cyc:{cyc}, Car:{car}'
            line = f'{f.stem} | {n_boxes:5d} | {top_score:.4f} | {breakdown}'
            lines.append(line)
            print(line)
        else:
            line = f'{f.stem} | error  | {type(out).__name__}'
            lines.append(line)
            print(line)

    lines.append('')
    lines.append(f'TOTALS:')
    lines.append(f'total_boxes: {total_boxes}')
    lines.append(f'pedestrian: {total_ped}')
    lines.append(f'cyclist: {total_cyc}')
    lines.append(f'car: {total_car}')
    lines.append(f'avg_boxes_per_frame: {total_boxes / len(files):.1f}')

    # Save to file
    out_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n✓ Saved to: {out_file.resolve()}')


if __name__ == '__main__':
    main()

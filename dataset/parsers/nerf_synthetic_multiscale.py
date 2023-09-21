from pathlib import Path
import numpy as np
from tqdm import tqdm

import dataset.utils.io as data_io
from dataset.utils.cameras import PinholeCamera


def load_data(base_path: Path, scene: str, split: str, cam_num: int = 4):
    # ipdb.set_trace()
    data_path = base_path / scene
    meta_path = data_path / 'metadata.json'

    splits = ['train', 'val'] if split == "trainval" else [split]
    meta = None
    for s in splits:
        m = data_io.load_from_json(meta_path)[s]
        if meta is None:
            meta = m
        else:
            for k, v in meta.items():
                v.extend(m[k])

    pix2cam = meta['pix2cam']
    poses = meta['cam2world']
    image_width = meta['width']
    image_height = meta['height']
    lossmult = meta['lossmult']

    assert image_height[0] == image_height[cam_num]
    assert image_width[0] == image_width[cam_num]
    assert pix2cam[0] == pix2cam[cam_num]
    assert lossmult[0] == lossmult[cam_num]
    cameras = []
    for i in range(cam_num):
        k = np.linalg.inv(pix2cam[i])
        fx = k[0, 0]
        fy = -k[1, 1]
        cx = -k[0, 2]
        cy = -k[1, 2]
        cam = PinholeCamera(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=image_width[i],
            height=image_height[i],
            # loss_multi=lossmult[i],
        )
        cameras.append(cam)

    frames = {k: [] for k in range(len(cameras))}
    index = 0
    for frame in tqdm(meta['file_path']):
        fname = data_path / frame
        frames[index % cam_num].append(
            {
                'image_filename': fname,
                'lossmult': lossmult[index],
            }
        )
        index = index + 1
    poses = {k: poses[k :: len(cameras)] for k in range(len(cameras))}

    aabb = np.array([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
    outputs = {
        'frames': frames,
        'poses': poses,
        'cameras': cameras,
        'aabb': aabb,
    }
    return outputs


if __name__ == '__main__':
    data = load_data(
        Path('/mnt/bn/wbhu-nerf/Dataset/nerf_synthetic_multiscale'),
        'lego',
        split='train',
    )
    pass

from pathlib import Path
from typing import List
import numpy as np
import scipy as sp
import torch

from rotation_conversions import axis_angle_to_matrix,  matrix_to_axis_angle



def random_vertical_rotations(n: int):
    angles = (torch.rand(n, dtype=torch.float32) * 2 * torch.pi).unsqueeze(1)
    axis = torch.tensor([0, 0, 1], dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    return axis_angle_to_matrix(axis * angles)



@torch.no_grad()
def prepare_data(splits, indir: Path, outdir: Path, fps: int, chunk_size: int, augment_count: int, stages):
    assert indir.exists(), f"Input directory {indir} does not exist"
    outdir.mkdir(parents=True, exist_ok=True)

    if 1 in stages:
        print(f"Stage 1: Downsampling and chunking data to {fps} fps and {chunk_size} frames")

        for split, datasets in splits.items():
            split_total_poses = []
            split_total_trans = []

            for data in datasets:
                files = indir.joinpath(data).glob('*/*_poses.npz')
                for file in files:
                    npz = np.load(file)
                    framerate = npz['mocap_framerate']
                    poses = torch.from_numpy(npz['poses'][:, :22 * 3]).to(torch.float32)
                    trans = torch.from_numpy(npz['trans']).to(torch.float32)

                    # Resample to target framerate
                    factor = framerate // fps
                    idx = torch.arange(0, len(poses), factor, dtype=torch.int)
                    poses = poses[idx]
                    trans = trans[idx]

                    # Convert from axis-angle to 6D rotation representation
                    poses =  axis_angle_to_matrix(poses.view(-1, 22, 3))

                    # Split into chunks
                    if len(poses) < chunk_size:
                        continue

                    n_chunks = len(poses) // chunk_size

                    idx = torch.randint(0, len(poses) - chunk_size + 1, (n_chunks,))

                    poses = torch.stack([poses[i:i + chunk_size] for i in idx])

                    trans = torch.stack([trans[i:i + chunk_size] for i in idx])

                    trans -= trans[:, 0 , :].unsqueeze(1).clone()
                    split_total_poses.append(poses)
                    split_total_trans.append(trans)

            split_total_poses = torch.cat(split_total_poses)
            split_total_trans = torch.cat(split_total_trans)


            print(split_total_poses.min(), split_total_poses.max(), split_total_poses.mean(), split_total_poses.std())
            print(split_total_trans.min(), split_total_trans.max(), split_total_trans.mean(), split_total_trans.std())

            torch.save({
                'poses': split_total_poses,
                'trans': split_total_trans
            }, outdir.joinpath(f'{split}.pt'))

    if 2 in stages:
        print("Stage 2: Augmenting trianing data with random rotations around vertical axis")

        total_poses = []
        total_trans = []
        for i in range(augment_count):
            train_data = torch.load(outdir.joinpath('train.pt'))

            poses = train_data['poses']
            trans = train_data['trans']

            rotations = random_vertical_rotations(poses.shape[0]).unsqueeze(1)

            root_orients = poses[:, :, 0, :, :]

            root_orients[:] = rotations @ root_orients

            trans = (rotations @ trans.unsqueeze(-1)).squeeze(-1)

            total_poses.append(poses)
            total_trans.append(trans)

        torch.save({
            'poses': matrix_to_axis_angle(torch.cat(total_poses)),
            'trans': torch.cat(total_trans), 
        }, outdir.joinpath(f'train.pt'))


        del total_poses
        del total_trans

        for split in ['val', 'test']:
            split_data = torch.load(outdir.joinpath(f'{split}.pt'))

            poses = split_data['poses']
            trans = split_data['trans']


            torch.save({
                'poses': matrix_to_axis_angle(poses),
                'trans': trans
            }, outdir.joinpath(f'{split}.pt'))
 




if __name__ == "__main__":
    splits = {
        'train': ['CMU'],
        'val': ['SFU'],
        'test': ['MPI_Limits']
    }

    indir = Path('/home/michaseyi/Downloads/mocap')
    outdir = Path('data_prepared')
    duration = 5
    fps = 15
    chunk_size = fps * duration
    augment_count = 4
    stages = [1, 2]
    
    prepare_data(splits, indir, outdir, fps, chunk_size, augment_count, stages)




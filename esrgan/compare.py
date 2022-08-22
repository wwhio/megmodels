import os
import os.path as osp
import numpy as np
import megengine
import megengine.module as M
import torch
import time

from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

import model
import model_pt


if megengine.is_cuda_available():
    megengine.set_default_device('gpu0')
else:
    megengine.set_default_device('cpu0')

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

rrdb_psnr_meg: M.Module = model.rrdb_psnr(pretrained=True)
rrdb_esrgan_meg: M.Module = model.rrdb_esrgan(pretrained=True)

rrdb_psnr_pt = model_pt.rrdb_psnr()
rrdb_esrgan_pt = model_pt.rrdb_esrgan()

rrdb_psnr_pt.load_state_dict(torch.load('../weights_pt/RRDB_PSNR_x4.pth'))
rrdb_esrgan_pt.load_state_dict(torch.load('../weights_pt/RRDB_ESRGAN_x4.pth'))

rrdb_psnr_pt.to(torch_device)
rrdb_esrgan_pt.to(torch_device)


def compare():
    rrdb_psnr_meg.eval()
    rrdb_esrgan_meg.eval()
    rrdb_psnr_pt.eval()
    rrdb_esrgan_pt.eval()
    torch.set_grad_enabled(False)

    set5_lr_dir = '../DATA/Set5/LRbicx4'
    set5_gt_dir = '../DATA/Set5/GTmod12'
    set5_files = [i for i in os.listdir(set5_lr_dir) if i.endswith('.png')]
    set5_files.sort()

    set14_lr_dir = '../DATA/Set14/LRbicx4'
    set14_gt_dir = '../DATA/Set14/GTmod12'
    set14_files = [i for i in os.listdir(set14_lr_dir) if i.endswith('.png')]
    set14_files.sort()

    print('Model Type: PSNR    Data: Set5')
    compare_psnr(set5_lr_dir, set5_gt_dir, set5_files)
    print('Model Type: PSNR    Data: Set14')
    compare_psnr(set14_lr_dir, set14_gt_dir, set14_files)

    print('Model Type: GAN    Data: Set5')
    compare_gan(set5_lr_dir, set5_gt_dir, set5_files)
    print('Model Type: GAN    Data: Set14')
    compare_gan(set14_lr_dir, set14_gt_dir, set14_files)


def compare_psnr(lr_dir, gt_dir, files, print_prefix='    '):
    idx = 0
    lr_files = [osp.join(lr_dir, i) for i in files]
    gt_files = [osp.join(gt_dir, i) for i in files]
    time_meg = 0
    time_torch = 0
    for lr_file, gt_file in zip(lr_files, gt_files):
        idx += 1

        gt_img = imread(gt_file)

        lr_img = imread(lr_file) / 255.0
        lr_inp = np.transpose(lr_img, (2, 0, 1)).astype(np.float32)  # c,h,w
        lr_inp = lr_inp[None, :, :, :]  # n,c,h,w
        lr_inp = np.copy(lr_inp)

        lr_meg = megengine.tensor(lr_inp)
        time0 = time.time()
        sr_meg = rrdb_psnr_meg(lr_meg).numpy()
        time_meg += time.time() - time0
        sr_meg = np.transpose(sr_meg[0, :, :, :], (1, 2, 0)).clip(0, 1)
        sr_meg = (sr_meg * 255.0).round().astype(gt_img.dtype)
        psnr_meg = peak_signal_noise_ratio(gt_img, sr_meg)

        lr_torch = torch.from_numpy(lr_inp).to(torch_device)
        time0 = time.time()
        sr_torch = rrdb_psnr_pt(lr_torch).cpu().numpy()
        time_torch += time.time() - time0
        sr_torch = np.transpose(sr_torch[0, :, :, :], (1, 2, 0)).clip(0, 1)
        sr_torch = (sr_torch * 255.0).round().astype(gt_img.dtype)
        psnr_torch = peak_signal_noise_ratio(gt_img, sr_torch)
        print(f'{print_prefix}ID{idx:3d}:'
              f'    PSNR MEG:{psnr_meg:8.4f}    PSNR TORCH:{psnr_torch:8.4f}'
              f'    PSNR DIFF: {psnr_meg-psnr_torch:.2f}')
    print(f'{print_prefix}TIME MEG:{time_meg:.2f}    TIME TORCH:{time_torch:.2f}')


def compare_gan(lr_dir, gt_dir, files, print_prefix='    '):
    idx = 0
    lr_files = [osp.join(lr_dir, i) for i in files]
    gt_files = [osp.join(gt_dir, i) for i in files]
    time_meg = 0
    time_torch = 0
    for lr_file, gt_file in zip(lr_files, gt_files):
        idx += 1

        gt_img = imread(gt_file)

        lr_img = imread(lr_file) / 255.0
        lr_inp = np.transpose(lr_img, (2, 0, 1)).astype(np.float32)  # c,h,w
        lr_inp = lr_inp[None, :, :, :]  # n,c,h,w
        lr_inp = np.copy(lr_inp)

        lr_meg = megengine.tensor(lr_inp)
        time0 = time.time()
        sr_meg = rrdb_esrgan_meg(lr_meg).numpy()
        time_meg += time.time() - time0
        sr_meg = np.transpose(sr_meg[0, :, :, :], (1, 2, 0)).clip(0, 1)
        sr_meg = (sr_meg * 255.0).round().astype(gt_img.dtype)
        psnr_meg = peak_signal_noise_ratio(gt_img, sr_meg)

        lr_torch = torch.from_numpy(lr_inp).to(torch_device)
        time0 = time.time()
        sr_torch = rrdb_esrgan_pt(lr_torch).cpu().numpy()
        time_torch += time.time() - time0
        sr_torch = np.transpose(sr_torch[0, :, :, :], (1, 2, 0)).clip(0, 1)
        sr_torch = (sr_torch * 255.0).round().astype(gt_img.dtype)
        psnr_torch = peak_signal_noise_ratio(gt_img, sr_torch)
        print(f'{print_prefix}ID{idx:3d}:'
              f'    PSNR MEG:{psnr_meg:8.4f}    PSNR TORCH:{psnr_torch:8.4f}'
              f'    ABS DIFF: {np.abs(sr_meg - sr_torch).sum()}')
    print(f'{print_prefix}TIME MEG:{time_meg:.2f}    TIME TORCH:{time_torch:.2f}')


if __name__ == '__main__':
    compare()

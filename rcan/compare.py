import glob
import numpy as np
import megengine
import torch

from skimage.io import imread

import model
import model_pt

from psnr import calculate_psnr

torch.set_grad_enabled(False)
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def release_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    megengine.coalesce_free_memory()


def compare():
    compare_x2()
    release_memory()
    compare_x3()
    release_memory()
    compare_x4()
    release_memory()


def compare_x2():
    print('Set 14:    degrad: bicubic    scale: x2')
    net_meg = model.rcan_x2(pretrained=True)
    net_meg.eval()

    net_torch = model_pt.rcan_x2()
    net_torch.load_state_dict(torch.load('../weights_pt/RCAN_BIX2.pt', map_location='cpu'))
    net_torch.to(torch_device)
    net_torch.eval()

    lr_files = glob.glob('../DATA/Set14/LRbicx2/*.png')
    lr_files.sort()
    hr_files = glob.glob('../DATA/Set14/GTmod12/*.png')
    hr_files.sort()
    file_list = list(zip(lr_files, hr_files))
    compare_inner(net_meg, net_torch, file_list)


def compare_x3():
    print('Set 14:    degrad: bicubic    scale: x3')
    net_meg = model.rcan_x3(pretrained=True)
    net_meg.eval()

    net_torch = model_pt.rcan_x3()
    net_torch.load_state_dict(torch.load('../weights_pt/RCAN_BIX3.pt', map_location='cpu'))
    net_torch.to(torch_device)
    net_torch.eval()

    lr_files = glob.glob('../DATA/Set14/LRbicx3/*.png')
    lr_files.sort()
    hr_files = glob.glob('../DATA/Set14/GTmod12/*.png')
    hr_files.sort()
    file_list = list(zip(lr_files, hr_files))
    compare_inner(net_meg, net_torch, file_list)


def compare_x4():
    print('Set 14:    degrad: bicubic    scale: x4')
    net_meg = model.rcan_x4(pretrained=True)
    net_meg.eval()

    net_torch = model_pt.rcan_x4()
    net_torch.load_state_dict(torch.load('../weights_pt/RCAN_BIX4.pt', map_location='cpu'))
    net_torch.to(torch_device)
    net_torch.eval()

    lr_files = glob.glob('../DATA/Set14/LRbicx4/*.png')
    lr_files.sort()
    hr_files = glob.glob('../DATA/Set14/GTmod12/*.png')
    hr_files.sort()
    file_list = list(zip(lr_files, hr_files))
    compare_inner(net_meg, net_torch, file_list)


def compare_inner(net_meg, net_torch, file_list):
    for idx, (lr, hr) in enumerate(file_list):
        hr_img = imread(hr)
        lr_img = imread(lr)
        lr_inp = np.transpose(lr_img, (2, 0, 1)).astype(np.float32)  # c,h,w
        lr_inp = lr_inp[None, :, :, :]
        lr_inp = np.copy(lr_inp)

        lr_meg = megengine.tensor(lr_inp)
        sr_meg = net_meg(lr_meg).numpy()
        sr_meg = np.transpose(sr_meg[0, :, :, :], (1, 2, 0)).clip(0, 255).round()
        psnr_meg = calculate_psnr(hr_img, sr_meg, border=8)

        lr_torch = torch.from_numpy(lr_inp).to(torch_device)
        sr_torch = net_torch(lr_torch).detach().cpu().numpy()
        sr_torch = np.transpose(sr_torch[0, :, :, :], (1, 2, 0)).clip(0, 255).round()
        psnr_torch = calculate_psnr(hr_img, sr_torch, border=8)

        print(f'    ID {idx:2d}:'
              f'    PSNR MEG: {psnr_meg:8.4f}'
              f'    PSNR TORCH: {psnr_torch:8.4f}'
              )


if __name__ == "__main__":
    compare()

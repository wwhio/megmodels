import os
import os.path
import time
import numpy as np
import megengine as meg
import torch

from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread

import model
import dncnn_pt

if meg.is_cuda_available():
    meg.set_default_device('gpu0')
else:
    meg.set_default_device('cpu0')

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

meg_model = model.dncnn_25(pretrained=True)

torch_model = dncnn_pt.DnCNN()
torch_model.load_state_dict(torch.load('../weights_pt/dncnn_pt.pt', map_location=torch.device('cpu')))
torch_model.to(torch_device)


def compare():
    meg_model.eval()
    torch_model.eval()
    torch.set_grad_enabled(False)

    data_folder = '../DATA/BSD68'
    data_file_list = sorted(
        [i for i in os.listdir(data_folder) if i.endswith('.png')])

    with ThreadPoolExecutor(max_workers=10) as pool:
        psnr_task = []
        ssim_task = []

        np.random.seed(seed=0)  # 可重复
        time0 = time.time()
        for img_path in data_file_list:
            x = np.array(imread(os.path.join(data_folder, img_path)), dtype=np.float32) / 255.0
            y = x + np.random.normal(0, 25 / 255.0, x.shape)  # 添加高斯噪声
            y = y.astype(np.float32)

            y_meg = meg.tensor(y).reshape(1, 1, y.shape[0], y.shape[1])
            x_hat_meg = meg_model(y_meg).numpy().reshape(y.shape[0], y.shape[1]).clip(0, 1)

            psnr_task.append(pool.submit(peak_signal_noise_ratio, x, x_hat_meg))
            ssim_task.append(pool.submit(structural_similarity, x, x_hat_meg))

        meg_psnr_list = [i.result() for i in psnr_task]
        meg_ssim_list = [i.result() for i in ssim_task]
        meg_time = time.time() - time0

        psnr_task = []
        ssim_task = []
        np.random.seed(seed=0)  # 可重复
        time0 = time.time()
        for img_path in data_file_list:
            x = np.array(imread(os.path.join(data_folder, img_path)), dtype=np.float32) / 255.0
            y = x + np.random.normal(0, 25 / 255.0, x.shape)  # 添加高斯噪声
            y = y.astype(np.float32)

            y_torch = torch.from_numpy(y).to(torch_device).view(1, 1, y.shape[0], y.shape[1])
            x_hat_torch = torch_model(y_torch).to('cpu').numpy().reshape(y.shape[0], y.shape[1]).clip(0, 1)
            psnr_task.append(pool.submit(peak_signal_noise_ratio, x, x_hat_torch))
            ssim_task.append(pool.submit(structural_similarity, x, x_hat_torch))
        torch_psnr_list = [i.result() for i in psnr_task]
        torch_ssim_list = [i.result() for i in ssim_task]
        torch_time = time.time() - time0

    for i, v in enumerate(zip(meg_psnr_list, torch_psnr_list)):
        print(f'PSNR {i:3d}:    MEG:{v[0]:8.4f}    TORCH:{v[1]:8.4f}')

    for i, v in enumerate(zip(meg_ssim_list, torch_ssim_list)):
        print(f'SSIM {i:3d}:    MEG:{v[0]:8.4f}    TORCH:{v[1]:8.4f}')

    print(f'TOTAL TIME:    MEG:{meg_time:8.4f}s   TORCH:{torch_time:8.4f}s')


if __name__ == '__main__':
    print('start comparing')
    compare()

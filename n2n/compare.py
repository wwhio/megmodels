import glob
import numpy as np
import megengine
import torch

from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread

import model
import model_pt


class Noise:
    def __init__(self, style):
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('-')]
            if len(self.params) == 1:
                self.style = 'gauss_fix'
            elif len(self.params) == 2:
                self.style = 'gauss_range'
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('-')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

        if self.style not in ['gauss_fix', 'gauss_range', 'poisson_fix', 'poisson_range']:
            raise ValueError(f'Unknown style: style = {style}')

    def __call__(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compare():
    image_paths = [i for i in glob.glob('../DATA/Set14_n2n/*.png')]
    image_paths.sort()

    torch.set_grad_enabled(False)

    np.random.seed(0)  # 可重复

    compare_gauss25(image_paths)
    compare_gauss5to50(image_paths)
    compare_poisson30(image_paths)
    compare_poisson5to50(image_paths)


def compare_gauss25(image_paths):
    print('Compare @ gauss 25')
    n2n = model.n2n_gauss25(pretrained=True)
    n2n.eval()

    n2n_pt = model_pt.neighbor2neighbor()
    n2n_pt.load_state_dict(torch.load('../weights_pt/n2n_model_gauss25_b4e100r02.pth'))
    n2n_pt.to(device=torch_device)
    n2n_pt.eval()

    noise_gen = Noise('gauss25')
    compare_inner(n2n, n2n_pt, noise_gen, image_paths)
    print('\n')


def compare_gauss5to50(image_paths):
    print('Compare @ gauss 5 ~ 50')
    n2n = model.n2n_gauss5to50(pretrained=True)
    n2n.eval()

    n2n_pt = model_pt.neighbor2neighbor()
    n2n_pt.load_state_dict(torch.load('../weights_pt/n2n_model_gauss5-50_b4e100r02.pth'))
    n2n_pt.to(device=torch_device)
    n2n_pt.eval()

    noise_gen = Noise('gauss5-50')
    compare_inner(n2n, n2n_pt, noise_gen, image_paths)
    print('\n')


def compare_poisson30(image_paths):
    print('Compare @ poisson 30')
    n2n = model.n2n_poisson30(pretrained=True)
    n2n.eval()

    n2n_pt = model_pt.neighbor2neighbor()
    n2n_pt.load_state_dict(torch.load('../weights_pt/n2n_model_poisson30_b4e100r02.pth'))
    n2n_pt.to(device=torch_device)
    n2n_pt.eval()

    noise_gen = Noise('poisson30')
    compare_inner(n2n, n2n_pt, noise_gen, image_paths)
    print('\n')


def compare_poisson5to50(image_paths):
    print('Compare @ poisson 5 ~ 50')
    n2n = model.n2n_poisson5to50(pretrained=True)
    n2n.eval()

    n2n_pt = model_pt.neighbor2neighbor()
    n2n_pt.load_state_dict(torch.load('../weights_pt/n2n_model_poisson5-50_b4e100r02.pth'))
    n2n_pt.to(device=torch_device)
    n2n_pt.eval()

    noise_gen = Noise('gauss5-50')
    compare_inner(n2n, n2n_pt, noise_gen, image_paths)
    print('\n')


def compare_inner(model, model_pt, noise_gen, image_paths):
    for idx, i in enumerate(image_paths):
        image_clean255 = imread(i)
        image_clean = image_clean255.astype(np.float32) / 255.0

        image_noise = noise_gen(image_clean)
        H, W = image_noise.shape[0], image_noise.shape[1]
        val_size = (max(H, W) + 32) // 32 * 32
        image_noise = np.pad(image_noise, [[0, val_size - H], [0, val_size - W], [0, 0]])
        image_noise = image_noise.transpose(2, 0, 1)[None, :, :, :]  # n,c,h,w

        image_clean_hat_pt = model_pt(torch.from_numpy(image_noise).to(device=torch_device)).cpu().numpy()[:, :, :H, :W]
        image_clean_hat_pt = image_clean_hat_pt[0, :, :, :].transpose(1, 2, 0)
        image_clean_hat_pt255 = (image_clean_hat_pt.clip(0, 1) * 255.0).round().astype(np.uint8)

        image_clean_hat = model(megengine.tensor(image_noise)).numpy()[:, :, :H, :W]
        image_clean_hat = image_clean_hat[0, :, :, :].transpose(1, 2, 0)
        image_clean_hat255 = (image_clean_hat.clip(0, 1) * 255.0).round().astype(np.uint8)

        psnr = peak_signal_noise_ratio(image_clean255, image_clean_hat255)
        psnr_pt = peak_signal_noise_ratio(image_clean255, image_clean_hat_pt255)

        print(f'    PSNR of {idx:2d}:    MEG:{psnr:8.4f}    TORCH:{psnr_pt:8.4f}')


if __name__ == '__main__':
    compare()

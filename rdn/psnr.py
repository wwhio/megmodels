import math
import numpy as np


def calculate_psnr(img1, img2, border=None):
    """calculate_psnr

    Args:
        img1 (ndarray): range [0, 255], order = HWC
        img2 (ndarray): range [0, 255], order = HWC
        border (int): crop border
    """

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    if border:
        img1 = img1[border:-border, border:-border, :]
        img2 = img2[border:-border, border:-border, :]

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0

    return 20.0 * math.log10(255.0 / math.sqrt(mse))

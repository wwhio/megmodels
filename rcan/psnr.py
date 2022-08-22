import math
import numpy as np


def calculate_psnr(img1, img2, border=0, test_y_channel=False):
    """calculate_psnr

    Args:
        img1 (ndarray): range [0, 255], order = HWC
        img2 (ndarray): range [0, 255], order = HWC
        border (int, optional): border to crop. Defaults to 0.
    """

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if border != 0:
        h, w = img1.shape[:2]
        img1 = img1[border:h - border, border:w - border]
        img2 = img2[border:h - border, border:w - border]

    if test_y_channel:
        img1 = rgb2ycbcr(img1, only_y=True)
        img2 = rgb2ycbcr(img2, only_y=True)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0

    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def rgb2ycbcr(img: np.ndarray, only_y=False):
    '''rgb2ycbcr
    Args:
        only_y: only return Y channel
        img: data range [0, 255]

    output: data range [0, 255]
    '''
    in_img_type = img.dtype
    img = img / 255.0  # data range [0, 1]

    # convert
    if only_y:
        # note: order HWC -> HW
        rlt = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) + [16, 128, 128]

    rlt = np.clip(rlt, 0, 255)
    if in_img_type == np.uint8:
        rlt = rlt.round()
    return rlt.astype(in_img_type)

import glob
import numpy
import megengine
import megengine.module
import torch

from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread

import model_csd_edsr_mix
import model_csd_edsr_student
import model_torch_csd_edsr_mix


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

csd_edsr_mix = model_csd_edsr_mix.csd_edsr_mix(pretrained=True)  # teacher and student in one model
csd_edsr_mix.eval()
csd_edsr_student = model_csd_edsr_student.csd_edsr_student(pretrained=True)  # student model
csd_edsr_student.eval()

csd_edsr_mix_torch = model_torch_csd_edsr_mix.csd_edsr_mix()  # teacher and student in one model
csd_edsr_mix_torch_state_dict = torch.load('../weights_pt/csd_edsr_x4_0.25student.pth')['model_state_dict']
csd_edsr_mix_torch_state_dict = {k[7:]: v for k, v in csd_edsr_mix_torch_state_dict.items()}
csd_edsr_mix_torch.load_state_dict(csd_edsr_mix_torch_state_dict)
csd_edsr_mix_torch.to(torch_device)
csd_edsr_mix_torch.eval()

torch.set_grad_enabled = False


def compare():
    lr_file_list = glob.glob('../DATA/Set14/LRbicx4/*.png')
    lr_file_list.sort()

    hr_file_list = glob.glob('../DATA/Set14/GTmod12/*.png')
    hr_file_list.sort()

    lr_hr_pair_list = list(zip(lr_file_list, hr_file_list))

    for idx, (lr, hr) in enumerate(lr_hr_pair_list):
        lr = imread(lr)
        hr = imread(hr)

        lr = lr.transpose(2, 0, 1)
        lr = lr[None, :, :, :].astype(numpy.float32)

        lr_meg = megengine.tensor(lr)
        lr_torch = torch.from_numpy(lr).to(torch_device)

        sr_teacher = csd_edsr_mix(lr_meg)
        sr_teacher = sr_teacher.numpy().clip(0, 255).round()
        sr_teacher = sr_teacher[0, :, :, :].transpose(1, 2, 0)
        sr_teacher = sr_teacher.astype(numpy.uint8)

        sr_teacher_torch = csd_edsr_mix_torch(lr_torch).detach().cpu()
        sr_teacher_torch = sr_teacher_torch.numpy().clip(0, 255).round()
        sr_teacher_torch = sr_teacher_torch[0, :, :, :].transpose(1, 2, 0)
        sr_teacher_torch = sr_teacher_torch.astype(numpy.uint8)

        sr_student = csd_edsr_student(lr_meg)
        sr_student = sr_student.numpy().clip(0, 255).round()
        sr_student = sr_student[0, :, :, :].transpose(1, 2, 0)
        sr_student = sr_student.astype(numpy.uint8)

        sr_mix = csd_edsr_mix(lr_meg, 0.25)  # student in mix model
        sr_mix = sr_mix.numpy().clip(0, 255).round()
        sr_mix = sr_mix[0, :, :, :].transpose(1, 2, 0)
        sr_mix = sr_mix.astype(numpy.uint8)

        sr_mix_torch = csd_edsr_mix_torch(lr_torch, 0.25).detach().cpu()
        sr_mix_torch = sr_mix_torch.numpy().clip(0, 255).round()
        sr_mix_torch = sr_mix_torch[0, :, :, :].transpose(1, 2, 0)
        sr_mix_torch = sr_mix_torch.astype(numpy.uint8)

        psnr_m = peak_signal_noise_ratio(hr, sr_teacher)
        psnr_t = peak_signal_noise_ratio(hr, sr_teacher_torch)
        psnr_m_stu = peak_signal_noise_ratio(hr, sr_student)
        psnr_m_mix = peak_signal_noise_ratio(hr, sr_mix)
        psnr_t_mix = peak_signal_noise_ratio(hr, sr_mix_torch)

        print(f"PSNR of {idx: 3d}")
        print(f"    MEG   (T): {psnr_m:7.4f}    MEG   (Stu in Mix): {psnr_m_mix:7.4f}    MEG (Stu): {psnr_m_stu:7.4f}")
        print(f"    TORCH (T): {psnr_t:7.4f}    TORCH (Stu in Mix): {psnr_t_mix:7.4f}")


if __name__ == '__main__':
    compare()

import cv2
import gc
import os
import piqa
import torch

from data import img_to_tensor
from losses import DISTSLoss

def main():
    hr_folder = r"X:\datasets\pbr\val\hr"
    sr_folder = r"X:\datasets\pbr\val\sr_nv"
    maps = ["diffuse", "normal", "roughness"]

    psnr_fn = piqa.PSNR().cuda().float()
    ssim_fn = piqa.SSIM().cuda().float()
    fsim_fn = piqa.FSIM().cuda().float()
    dists_fn = DISTSLoss().cuda().bfloat16()

    for mat in os.listdir(hr_folder):
        for map_ in maps:
            torch.cuda.empty_cache()
            gc.collect()
            
            hr_path = os.path.join(hr_folder, mat, f"{map_}.png")
            sr_path = os.path.join(sr_folder, mat, f"{map_}.png")

            hr = img_to_tensor(cv2.imread(hr_path, cv2.IMREAD_COLOR)).unsqueeze(0).cuda().float()
            sr = img_to_tensor(cv2.imread(sr_path, cv2.IMREAD_COLOR)).unsqueeze(0).cuda().float()

            psnr = psnr_fn(sr, hr)
            ssim = ssim_fn(sr, hr)
            fsim = fsim_fn(sr, hr)
            dists = dists_fn(sr.bfloat16(), hr.bfloat16(), require_grad=False)

            print(f"{mat} {map_}: PSNR: {psnr:.2f}, SSIM: {ssim:.3f}, FSIM: {fsim:.3f}, DISTS: {dists:.3f}")


if __name__ == "__main__":
    main()
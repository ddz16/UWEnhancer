import argparse
import cv2
import glob
import numpy as np
import os.path as osp

from basicsr.utils import img2tensor
from basicsr.metrics import calculate_psnr, calculate_ssim

# import lpips


def main(args):
    # loss_fn_vgg = lpips.LPIPS(net='vgg', pnet_rand=False, model_path='experiments/pretrained_models/vgg16-397923af.pth').cuda()  # RGB, normalized to [-1,1]
    # lpips_all = []
    psnr_all = []
    ssim_all = []
    img_list = sorted(glob.glob(osp.join(args.restored, '*')))

    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_restored = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_gt = cv2.imread(osp.join(args.gt, basename + ext), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_gt = cv2.resize(img_gt, (256, 256))

        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        psnr_all.append(psnr)
        ssim_all.append(ssim)

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)

        # calculate lpips
        # lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda(), normalize=True).item()

        print(f'{i+1:3d}: {basename:20}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}.')
        # lpips_all.append(lpips_val)

    print(args.restored)
    print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')
    # print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='datasets/val_set14/Set14', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, default='results/Set14', help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    args = parser.parse_args()
    main(args)

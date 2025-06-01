import argparse
import cv2
import os
import warnings

from basicsr.metrics import calculate_niqe, calculate_piqe, calculate_uciqe, calculate_uiqm
from basicsr.utils import scandir


def main(args):
    niqe_all = []
    piqe_all = []
    uciqe_all = []
    uiqm_all = []
    img_list = sorted(scandir(args.input, recursive=True, full_path=True))

    for i, img_path in enumerate(img_list):
        basename, _ = os.path.splitext(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_score = calculate_niqe(img, args.crop_border, input_order='HWC', convert_to='y')
            piqe_score = calculate_piqe(img, args.crop_border)
            uciqe_score = calculate_uciqe(img, args.crop_border)
            uiqm_score = calculate_uiqm(img, args.crop_border)

        print(f'{i+1:3d}: {basename:20}. \tNIQE: {niqe_score:.6f}, \tPIQE: {piqe_score:.6f}, \tUCIQE: {uciqe_score:.6f}, \tUIQM: {uiqm_score:.6f}')
        niqe_all.append(niqe_score)
        piqe_all.append(piqe_score)
        uciqe_all.append(uciqe_score)
        uiqm_all.append(uiqm_score)
        
    print(args.input)
    print(f'Average: NIQE: {sum(niqe_all) / len(niqe_all):.6f}')
    print(f'Average: PIQE: {sum(piqe_all) / len(piqe_all):.6f}')
    print(f'Average: UCIQE: {sum(uciqe_all) / len(uciqe_all):.6f}')
    print(f'Average: UIQM: {sum(uiqm_all) / len(uiqm_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='results/Fusion/C60', help='Input path')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    args = parser.parse_args()
    main(args)

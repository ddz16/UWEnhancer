import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

from basicsr.archs.uwenhancer_arch import UWEnhancer
from basicsr.utils.img_util import img2tensor, tensor2img

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='inference/')
    parser.add_argument(
        '--model_path',
        type=str,
        default= 'experiments/08_UWEnhancer_LSUI_in7_lr03_lossw02/models/net_g_latest.pth')
    args = parser.parse_args()
    result_root = 'results/000000/'
    os.makedirs(result_root, exist_ok=True)

    # set up the RIDNet
    net = UWEnhancer().to(device)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['params'])
    net.eval()

    # scan all the jpg and png images
    img_list = sorted(glob.glob(os.path.join(args.test_path, '*.[jp][pn]g')))
    pbar = tqdm(total=len(img_list), desc='')
    for idx, img_path in enumerate(img_list):
        img_name = os.path.basename(img_path)
        print(img_name)
        pbar.update(1)
        pbar.set_description(f'{idx}: {img_name}')
        # read image
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        img = cv2.resize(img, (256, 256)) 
        img = img2tensor(img / 255.0, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
        # inference
        with torch.no_grad():
            _, output, _, _, _, _, _ = net(img)
        # save image
        output = tensor2img(output)
        save_img_path = os.path.join(result_root, img_name)
        output = cv2.resize(output, (width, height))
        cv2.imwrite(save_img_path, output)

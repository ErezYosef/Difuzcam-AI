import argparse
import glob
import logging
import os
import sys
import torchvision.utils

from numpy import log10
import torch

from time import gmtime, strftime, localtime
from tqdm import tqdm
import torch.nn as nn

sys.path.append('.')

import yaml
import numpy as np
def dict_representor_yaml(dict_in):
    out = {}
    for k,v in dict_in.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            data = v.tolist()
        elif isinstance(v,dict):
            data = dict_representor_yaml(v)
        else:
            data = v
        out[k] = data
    return out



from PIL import Image



def eval_saved(load_path_base):
    metrics_dict={}
    load_path = os.path.join(load_path_base, 'save_results')
    gt_files = glob.glob(os.path.join(load_path, 'gt*.pt'))
    samples_files = glob.glob(os.path.join(load_path, 'sample*.pt'))

    gt_files.sort()
    samples_files.sort()

    lossf = nn.MSELoss(reduction='none')
    total_samples=0
    tot_gts = 0

    for i ,(gt_path, sample_path) in enumerate(zip(gt_files, samples_files)):
        imname = os.path.basename(sample_path)
        gt_img = torch.load(gt_path, map_location='cpu')
        sample_image = torch.load(sample_path, map_location='cpu')

        gt_img = (gt_img / 2 + 0.5).clamp(0, 1).cuda()
        sample_image = (sample_image / 2 + 0.5).clamp(0, 1).cuda()

        resize = torchvision.transforms.Resize((256, 256), antialias=True)
        gt_img = resize(gt_img)
        sample_image = resize(sample_image)

        mse_loss_list = []

        for i in range(sample_image.shape[0]):
            sample_image_i = sample_image[i].unsqueeze(0)
            mse_loss = torch.mean(lossf(sample_image_i, gt_img), (1, 2, 3))

            mse_loss_list.append(mse_loss.item())
            total_samples += 1

        metrics_dict[imname] = {'mse': mse_loss_list}
        tot_gts += 1

    tot_loss = torch.tensor([v['mse'][0] for k,v in metrics_dict.items()])

    tst_score = tot_loss.mean().item()
    tot_psnr = (10 * torch.log10(1 / tot_loss)).mean().item()
    print(f'Test loss/PSNR: {tst_score:.5g} / {tot_psnr:.5g}')

    metrics_dict['mean_loss'] = tst_score
    metrics_dict['psnr'] = tot_psnr

    with open(os.path.join(load_path_base, f'eval_saved_resized.yaml'), 'w') as outfile:
        yaml.dump(dict_representor_yaml(metrics_dict), outfile, indent=4)

    return dict_representor_yaml(metrics_dict)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--folder', dest='folder', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-t', '--tag', dest='tag', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('--config-file', dest='config_file', default='test_config.yaml',
                        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.tag is not None:
        load_path_base = os.path.join('<RESULTS_DIR>', args.tag)
        load_path_base = sorted(glob.glob(load_path_base))
        selected = -1
        load_path_base = load_path_base[selected]

    else:
        load_path_base = args.folder

    result = eval_saved(load_path_base)


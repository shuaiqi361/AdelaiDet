# coding:utf-8

import os
import cv2
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

from MaskLoader import MaskLoader
from utils import (
    IOUMetric,
    fast_ista,
    prepare_distance_transform_from_mask,
    prepare_overlay_DTMs_from_mask,
    prepare_extended_DTMs_from_mask
)

sns.set_theme()
sns.set(font_scale=2.4)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation for Sparse Mask Encoding with DTMs.')
    parser.add_argument('--data_root', default='/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17',
                        type=str)
    parser.add_argument('--dataset', default='coco_2017_val', type=str)
    parser.add_argument('--dictionary', default='/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/'
                                                'sparse_shape_dict/mask_fromDTM_minusone_basis_m28_n64_a0.50.npy',
                        type=str)
    # parser.add_argument('--dictionary', default='/media/keyi/Data/Research/traffic/detection/AdelaiDet/adet/modeling/'
    #                                             'DTInst/dictionary/Basis_L2_DTMs_overlay_m28_n256_a0.10.npy',
    #                     type=str)
    # mask encoding params.
    parser.add_argument('--mask_size', default=28, type=int)
    parser.add_argument('--n_codes', default=64, type=int)
    parser.add_argument('--n_vertices', default=360, type=int)
    parser.add_argument('--sparse_alpha', default=0.30, type=float)
    parser.add_argument('--batch-size', default=1000, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # parse args.
    mask_size = args.mask_size
    n_codes = args.n_codes
    n_vertices = args.n_vertices
    sparse_alpha = args.sparse_alpha

    dataset_root = args.data_root
    dictionary_path = args.dictionary

    # load matrix.
    print("Loading matrix parameters: {}".format(dictionary_path.split('/')[-1]))
    learned_dict = torch.from_numpy(np.load(dictionary_path)).to(torch.float32)

    # build data loader.
    mask_data = MaskLoader(root=dataset_root, dataset=args.dataset, size=mask_size)
    mask_loader = DataLoader(mask_data, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    size_data = len(mask_loader)
    sparsity_counts = []
    codes = []

    # evaluation.
    IoUevaluate = IOUMetric(2)
    print("Start evaluation ...")
    for i, masks in enumerate(mask_loader):
        print("Eva [{} / {}]".format(i, size_data))
        # generate the reconstruction mask.
        masks = masks.view(masks.shape[0], -1)  # a batch of masks: (N, 784)
        masks = masks.to(torch.float32)
        dtms = prepare_distance_transform_from_mask(masks, mask_size)  # for learn DTMs minusone
        # dtms = prepare_overlay_DTMs_from_mask(masks, mask_size)  # for learn DTMs overlay
        # dtms = prepare_extended_DTMs_from_mask(masks, mask_size)  # for learn DTMs extended, 0, 1 range

        # --> encode --> decode.
        dtms_codes = fast_ista(dtms, learned_dict, lmbda=sparse_alpha, max_iter=100)
        dtms_rc = torch.matmul(dtms_codes, learned_dict).numpy()

        # evaluate sparsity
        sparsity_counts.append(np.sum(np.abs(dtms_codes.numpy()) > 1e-4))

        # eva.
        dtms_rc = np.where(dtms_rc + 1 - 0.1 > 0.5, 1, 0)  # adjust the thresholding to binary masks
        # dtms_rc = np.where(dtms_rc - 0.1 > 0.5, 1, 0)
        # dtms_rc = np.where(dtms_rc > 0.4, 1, 0)
        IoUevaluate.add_batch(dtms_rc, masks.numpy())
        codes.append(dtms_codes.numpy())

        # do some plots
        for plot_data in dtms_codes.numpy():
            ax = sns.heatmap(np.abs(plot_data.reshape((8, 8))), linewidths=.5)

            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            # cbar = ax.colorbar(surf, shrink=0.8, aspect=20, fraction=.12, pad=.02)
            # cbar.set_label('Magnitude', size=16)
            # # access to cbar tick labels:
            # cbar.ax.tick_params(labelsize=20)

            plt.show()
        # break

    # codes = np.concatenate(codes, axis=0)  # n_data, n_codes
    # # plot heatmap for active coefficeints, show frequency
    # # plot_data = np.sum(np.abs(codes), axis=0) / len(codes)
    # plot_data = np.abs(codes[0])
    # print(plot_data.shape, len(codes))
    # # ax = sns.heatmap(plot_data.reshape((8, 8)), linewidths=.5, annot=True, fmt=".2f")
    # ax = sns.heatmap(plot_data.reshape((8, 8)), linewidths=.5, fontsize=26)
    #
    # plt.show()
    # exit()

    _, _, _, mean_iu, _ = IoUevaluate.evaluate()
    print("The mIoU for {}: {}".format(dictionary_path.split('/')[-1], mean_iu))
    print('Overall code activation rate: ', np.sum(sparsity_counts) * 1. / size_data / n_codes / args.batch_size)

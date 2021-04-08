# coding:utf-8

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import kurtosis
from MaskLoader import MaskLoader
from utils import (
    IOUMetric,
    transform,
    inverse_transform,
    direct_sigmoid,
    inverse_sigmoid,
    fast_ista
)


VALUE_MAX = 0.05
VALUE_MIN = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation for Sparse Mask Encoding.')
    parser.add_argument('--data_root', default='/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17',
                        type=str)
    parser.add_argument('--dataset', default='coco_2017_val', type=str)
    parser.add_argument('--dictionary', default='/media/keyi/Data/Research/traffic/detection/AdelaiDet/adet/'
                                                'modeling/SMInst/dictionary/mask_fromMask_basis_m28_n384_a0.40.npy',
                        type=str)
    # mask encoding params.
    parser.add_argument('--mask_size', default=28, type=int)
    parser.add_argument('--n_codes', default=384, type=int)
    parser.add_argument('--mask_sparse_alpha', default=0.40, type=float)
    parser.add_argument('--batch-size', default=1000, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # parse args.
    mask_size = args.mask_size
    n_codes = args.n_codes
    sparse_alpha = args.mask_sparse_alpha
    # class_agnostic = args.class_agnostic
    # whiten = args.whiten
    # sigmoid = args.sigmoid

    # cur_path = os.path.abspath(os.path.dirname(__file__))
    # root_path = args.root
    dataset_root = args.data_root
    dictionary_path = args.dictionary

    # load matrix.
    print("Loading matrix parameters: {}".format(dictionary_path.split('/')[-1]))
    learned_dict = torch.from_numpy(np.load(dictionary_path)).to(torch.float32)

    # build data loader.
    mask_data = MaskLoader(root=dataset_root, dataset=args.dataset, size=mask_size)
    mask_loader = DataLoader(mask_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    size_data = len(mask_loader)
    sparsity_counts = []
    kurtosis_counts = []

    # evaluation.
    IoUevaluate = IOUMetric(2)
    print("Start evaluation ...")
    for i, masks in enumerate(mask_loader):
        print("Eva [{} / {}]".format(i, size_data))
        # generate the reconstruction mask.
        masks = masks.view(masks.shape[0], -1)  # a batch of masks: (N, 784)
        masks = masks.to(torch.float32)

        # --> encode --> decode.
        mask_codes = fast_ista(masks, learned_dict, lmbda=sparse_alpha, max_iter=70)
        mask_rc = torch.matmul(mask_codes, learned_dict).numpy()

        sparsity_counts.append(np.mean(np.abs(mask_codes.numpy()) > 1e-2, axis=1))
        kurtosis_counts.append(mask_codes.numpy())

        # eva.
        mask_rc = np.where(mask_rc >= 0.5, 1, 0)
        IoUevaluate.add_batch(mask_rc, masks.numpy())

    _, _, _, mean_iu, _ = IoUevaluate.evaluate()
    print("The mIoU for {}: {}".format(dictionary_path.split('/')[-1], mean_iu))
    sparsity_counts = np.concatenate(sparsity_counts, axis=0)
    print('Overall code activation rate: ', np.mean(sparsity_counts))
    # print('Overall code activation rate: ', np.sum(sparsity_counts) * 1. / size_data / n_codes / args.batch_size)

    # calculate Kurtosis for predicted codes
    kurtosis_counts = np.concatenate(kurtosis_counts, axis=0)
    abs_codes = kurtosis_counts ** 2.
    kur = np.sum(kurtosis(kurtosis_counts, axis=1, fisher=True, bias=False)) / len(kurtosis_counts)
    print('Overall Kurtosis: ', kur)

    # calculate the variance explained by the top 60 codes
    var_explained = []
    for i in range(abs_codes.shape[0]):
        t_ = np.sort(abs_codes[i])[::-1]
        var_explained.append(np.sum(t_[:60]) / (np.sum(t_) + 1e-6))

    print('Variance explained: ', np.mean(var_explained))

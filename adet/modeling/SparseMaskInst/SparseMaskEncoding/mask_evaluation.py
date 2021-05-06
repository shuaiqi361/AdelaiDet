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
    fast_ista
)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation for Sparse Mask Encoding.')
    parser.add_argument('--data_root', default='/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17',
                        type=str)
    parser.add_argument('--dataset', default='coco_2017_val', type=str)
    parser.add_argument('--dictionary', default='/media/keyi/Data/Research/traffic/detection/AdelaiDet/adet/'
                                                'modeling/SparseMaskInst/dictionary/Centered_whitened_mask_basis_m28_n128_a0.20.npz',
                        type=str)
    # mask encoding params.
    parser.add_argument('--mask_size', default=28, type=int)
    parser.add_argument('--n_codes', default=128, type=int)
    parser.add_argument('--mask_sparse_alpha', default=0.2, type=float)
    parser.add_argument('--batch-size', default=1000, type=int)
    parser.add_argument('--top-code', default=60, type=int)
    parser.add_argument('--if-whiten', default=True, type=bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # parse args.
    mask_size = args.mask_size
    n_codes = args.n_codes
    sparse_alpha = args.mask_sparse_alpha

    dataset_root = args.data_root
    dictionary_path = args.dictionary

    # load matrix.
    print("Loading matrix parameters: {}".format(dictionary_path.split('/')[-1]))
    parameters = np.load(dictionary_path)
    learned_dict = parameters['shape_basis']
    shape_mean = parameters['shape_mean']
    shape_std = parameters['shape_std']
    learned_dict = torch.from_numpy(learned_dict).to(torch.float32)
    shape_mean = torch.from_numpy(shape_mean).to(torch.float32)
    shape_std = torch.from_numpy(shape_std).to(torch.float32)

    # build data loader.
    mask_data = MaskLoader(root=dataset_root, dataset=args.dataset, size=mask_size)
    mask_loader = DataLoader(mask_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    size_data = len(mask_loader)
    sparsity_counts = []
    kurtosis_counts = []
    all_masks = []
    reconstruction_error = []

    # evaluation.
    IoUevaluate = IOUMetric(2)
    print("Start evaluation ...")
    for i, masks in enumerate(mask_loader):
        print("Eva [{} / {}]".format(i, size_data))
        # generate the reconstruction mask.
        masks = masks.view(masks.shape[0], -1)  # a batch of masks: (N, 784)
        masks = masks.to(torch.float32)
        all_masks.append(masks)

        # --> encode --> decode.
        if args.if_whiten:
            centered_masks = (masks - shape_mean) / shape_std
            mask_codes = fast_ista(centered_masks, learned_dict, lmbda=sparse_alpha, max_iter=80)
            mask_rc = torch.matmul(mask_codes, learned_dict) * shape_std + shape_mean
            mask_rc = mask_rc.numpy()
        else:
            centered_masks = masks - shape_mean
            mask_codes = fast_ista(centered_masks, learned_dict, lmbda=sparse_alpha, max_iter=80)
            mask_rc = torch.matmul(mask_codes, learned_dict) + shape_mean
            mask_rc = mask_rc.numpy()

        sparsity_counts.append(np.mean(np.abs(mask_codes.numpy()) > 1e-2, axis=1))
        kurtosis_counts.append(mask_codes.numpy())

        # eva.
        mask_rc = np.where(mask_rc >= 0.5, 1, 0)
        IoUevaluate.add_batch(mask_rc, masks.numpy())
        # break

    _, _, _, mean_iu, _ = IoUevaluate.evaluate()
    print("The mIoU for {}: {}".format(dictionary_path.split('/')[-1], mean_iu))
    sparsity_counts = np.concatenate(sparsity_counts, axis=0)
    print('Overall code activation rate: ', np.mean(sparsity_counts))

    # calculate Kurtosis for predicted codes
    kurtosis_counts = np.concatenate(kurtosis_counts, axis=0)
    all_mask_codes = kurtosis_counts.copy()
    abs_codes = kurtosis_counts ** 2.
    kur = np.sum(kurtosis(kurtosis_counts, axis=1, fisher=True, bias=False)) / len(kurtosis_counts)
    print('Overall Kurtosis: ', kur)

    # calculate the variance explained by the top 60 codes
    # all_masks = np.concatenate(all_masks, axis=0)
    # print('Total number of instances evaluated: ', all_masks.shape)
    # total_var = np.sum(np.var(all_masks, axis=0))
    # var_explained = []
    # print('For each shape: ')
    # learned_dict = learned_dict.numpy()
    # for i in range(all_mask_codes.shape[0]):
    #     idx_codes = np.argsort(abs_codes[i])[::-1][:args.top_code]
    #     # print('idx shape: ', idx_codes.shape)
    #     mask_code_ = np.zeros(shape=all_mask_codes[i].shape)
    #     mask_code_[idx_codes] = 1
    #     # print('mask_code_ shape: ', mask_code_.shape)
    #     rec_error = all_masks[i, :] - np.matmul(all_mask_codes[i] * mask_code_, learned_dict)  # keep the top-60 components
    #     # print('rec_error shape: ', rec_error.shape)
    #     rec_error = np.sum(rec_error ** 2)
    #     # exit()
    #
    #     var_explained.append(1 - rec_error / total_var)
    #
    # print('Total variance: ', total_var)
    # print('Variance explained: ', np.mean(var_explained))

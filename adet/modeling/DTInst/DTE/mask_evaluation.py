# coding:utf-8

import os
import cv2
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import kurtosis
from MaskLoader import MaskLoader
from utils import (
    IOUMetric,
    fast_ista,
    prepare_distance_transform_from_mask,
    prepare_overlay_DTMs_from_mask,
    prepare_extended_DTMs_from_mask,
    prepare_augmented_distance_transform_from_mask,
    prepare_distance_transform_from_mask_with_weights,
    prepare_complement_distance_transform_from_mask_with_weights
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation for Sparse Mask Encoding with DTMs.')
    parser.add_argument('--data_root', default='/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17',
                        type=str)
    parser.add_argument('--dataset', default='coco_2017_val', type=str)
    # parser.add_argument('--dictionary', default='/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/'
    #                                             'sparse_shape_dict/mask_fromDTM_minusone_basis_m28_n256_a0.30.npy',
    #                     type=str)
    parser.add_argument('--dictionary', default='/media/keyi/Data/Research/traffic/detection/AdelaiDet/adet/modeling/'
                                                'DTInst/dictionary/mask_fromDTM_complement_basis_m28_n256_a0.30.npy',
                        type=str)
    # mask encoding params.
    parser.add_argument('--mask_size', default=28, type=int)
    parser.add_argument('--n_codes', default=256, type=int)
    parser.add_argument('--n_vertices', default=360, type=int)
    parser.add_argument('--sparse_alpha', default=0.30, type=float)
    parser.add_argument('--batch-size', default=1000, type=int)
    parser.add_argument('--top-code', default=60, type=int)
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
        # dtms = prepare_distance_transform_from_mask(masks, mask_size)  # for learn DTMs minusone
        # dtms = prepare_augmented_distance_transform_from_mask(masks, mask_size)
        # dtms = prepare_overlay_DTMs_from_mask(masks, mask_size)  # for learn DTMs overlay
        # dtms = prepare_extended_DTMs_from_mask(masks, mask_size)  # for learn DTMs extended, 0, 1 range
        # dtms, _, _ = prepare_distance_transform_from_mask_with_weights(masks, mask_size)
        dtms, _, _ = prepare_complement_distance_transform_from_mask_with_weights(masks, mask_size)
        all_masks.append(dtms)

        # --> encode --> decode.
        dtms_codes = fast_ista(dtms, learned_dict, lmbda=sparse_alpha, max_iter=80)
        dtms_rc = torch.matmul(dtms_codes, learned_dict).numpy()

        # evaluate sparsity
        sparsity_counts.append(np.sum(np.abs(dtms_codes.numpy()) > 1e-4))
        kurtosis_counts.append(dtms_codes.numpy())

        # eva.
        # dtms_rc = np.where(dtms_rc + 1 - 0.35 > 0.5, 1, 0)  # adjust the thresholding to binary masks
        # dtms_rc = np.where(dtms_rc + 0.9 > 0.5, 1, 0)
        # dtms_rc = np.where((dtms_rc + 1.) / 2 + 0.15 > 0.5, 1, 0)
        # dtms_rc = np.where(dtms_rc + 0.5 >= 0.5, 1, 0)
        dtms_rc = np.where(dtms_rc + 0.55 >= 0.5, 1, 0)
        IoUevaluate.add_batch(dtms_rc, masks.numpy())

    _, _, _, mean_iu, _ = IoUevaluate.evaluate()
    print("The mIoU for {}: {}".format(dictionary_path.split('/')[-1], mean_iu))
    print('Overall code activation rate: ', np.sum(sparsity_counts) * 1. / size_data / n_codes / args.batch_size)

    # # calculate Kurtosis for predicted codes
    # kurtosis_counts = np.concatenate(kurtosis_counts, axis=0)
    # all_mask_codes = kurtosis_counts.copy()
    # abs_codes = kurtosis_counts ** 2.
    # kur = np.sum(kurtosis(kurtosis_counts, axis=1, fisher=True, bias=False)) / len(kurtosis_counts)
    # print('Overall Kurtosis: ', kur)
    #
    # # calculate the variance explained by the top 60 codes
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
    #     rec_error = all_masks[i, :] - np.matmul(all_mask_codes[i] * mask_code_,
    #                                             learned_dict)  # keep the top-60 components
    #     # print('rec_error shape: ', rec_error.shape)
    #     rec_error = np.sum(rec_error ** 2)
    #     # exit()
    #
    #     var_explained.append(1 - rec_error / total_var)
    #
    # print('Total variance: ', total_var)
    # print('Variance explained: ', np.mean(var_explained))


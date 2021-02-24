# coding:utf-8

import os
import cv2
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from MaskLoader import MaskLoader
from utils import (
    IOUMetric,
    fast_ista,
    prepare_polygon_from_mask,
    poly_to_mask
)


VALUE_MAX = 0.05
VALUE_MIN = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation for Sparse Mask Encoding.')
    parser.add_argument('--data_root', default='/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17',
                        type=str)
    parser.add_argument('--dataset', default='coco_2017_val', type=str)
    parser.add_argument('--dictionary', default='/media/keyi/Data/Research/traffic/detection/AdelaiDet/adet/modeling/'
                                                'SSInst/dictionary/multi_fromMask_dict_m40_n64_v360_a0.01.npy',
                        type=str)
    # mask encoding params.
    parser.add_argument('--mask_size', default=40, type=int)
    parser.add_argument('--n_codes', default=64, type=int)
    parser.add_argument('--n_vertices', default=360, type=int)
    parser.add_argument('--sparse_alpha', default=0.01, type=float)
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
    mask_loader = DataLoader(mask_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    size_data = len(mask_loader)

    # evaluation.
    IoUevaluate = IOUMetric(2)
    print("Start evaluation ...")
    for i, masks in enumerate(mask_loader):
        print("Eva [{} / {}]".format(i, size_data))
        # generate the reconstruction mask.
        masks = masks.view(masks.shape[0], -1)  # a batch of masks: (N, 1600)
        polys = prepare_polygon_from_mask(masks, mask_size, n_vertices, pads=5)
        # masks = masks.to(torch.float32)

        # --> encode --> decode.
        polygon_codes = fast_ista(polys, learned_dict, lmbda=sparse_alpha, max_iter=80)
        polygon_rc = torch.matmul(polygon_codes, learned_dict)
        # eva.
        mask_rc = poly_to_mask(polygon_rc, mask_size)
        # mask_rc = poly_to_mask(polys, mask_size)
        # for j in range(mask_rc.shape[0]):
        #     show_img = np.concatenate([masks[j].numpy().reshape((mask_size, mask_size)),
        #                                mask_rc[j].numpy().reshape((mask_size, mask_size))],
        #                               axis=1).astype(np.uint8) * 255
        #     cv2.imshow('cat', show_img)
        #     if cv2.waitKey() & 0xFF == ord('q'):
        #         break
        IoUevaluate.add_batch(mask_rc.numpy(), masks.numpy())

    _, _, _, mean_iu, _ = IoUevaluate.evaluate()
    print("The mIoU for {}: {}".format(dictionary_path.split('/')[-1], mean_iu))

# coding:utf-8

import os
import json
import numpy as np

import torch.utils.data as data

from detectron2.structures import (
    Boxes,
    PolygonMasks,
    BoxMode
)


DATASETS = {
        "cityscapes_train": {
            "img_dir": "leftImg8bit_trainvaltest/leftImg8bit/train",
            "ann_dir": "gtFine/train"
        },
        "cityscapes_val": {
            "img_dir": "leftImg8bit_trainvaltest/leftImg8bit/val",
            "ann_dir": "gtFine/val"
        },
        "cityscapes_test": {
            "img_dir": "leftImg8bit_trainvaltest/leftImg8bit/test",
            "ann_dir": "gtFine/test"
        },
}

CITYSCAPES_CLASSES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


class CityscalesMaskLoader(data.Dataset):
    """
    Dataloader for Local Mask.

    Arguments:
        root (string): filepath to dataset folder.
        dataset (string): mask to use (eg. 'train', 'val').
        size (tuple): The size used for train/val (height, width).
        transform (callable, optional): transformation to perform on the input mask.

    """

    def __init__(self, root="/media/keyi/Data/Research/traffic/data/cityscapes", split="val", size=28):
        self.root = root
        self.split = split

        if isinstance(size, int):
            self.size = size
        else:
            raise TypeError

        data_info = DATASETS['cityscapes_' + split]
        img_dir, ann_dir = data_info['img_dir'], data_info['ann_dir']
        img_dir = os.path.join(self.root, img_dir)  # actually we do not use it.
        ann_dir = os.path.join(self.root, ann_dir)

        self.all_annotations = list()
        img_scene_list = os.listdir(img_dir)

        for scene in img_scene_list:
            print('process scene: ', scene)
            image_folder = os.path.join(img_dir, scene)
            all_images = os.listdir(image_folder)

            for img_name in all_images:
                if not img_name.endswith('.png'):
                    continue
                ann_folder = os.path.join(ann_dir, scene)
                ann_name = img_name.split('_')
                ann_name = ann_name[0] + '_' + ann_name[1] + '_' + ann_name[2] + '_gtFine_polygons.json'
                ann_file = os.path.join(ann_folder, ann_name)

                with open(ann_file, 'r') as f_ann:
                    ann_json = json.load(f_ann)

                h_img = int(ann_json['imgHeight'])
                w_img = int(ann_json['imgWidth'])
                if h_img < 2 or w_img < 2:
                    continue
                all_objects = ann_json['objects']
                for obj in all_objects:

                    cat_name = obj['label']
                    if cat_name not in CITYSCAPES_CLASSES:
                        continue

                    instance_polygon = np.array(obj['polygon'])
                    instance_polygon[:, 0] = np.clip(instance_polygon[:, 0], 0, w_img - 1)
                    instance_polygon[:, 1] = np.clip(instance_polygon[:, 1], 0, h_img - 1)

                    gt_x1, gt_y1, gt_x2, gt_y2 = int(min(instance_polygon[:, 0])), int(min(instance_polygon[:, 1])), \
                                                 int(max(instance_polygon[:, 0])), int(max(instance_polygon[:, 1]))

                    obj_ann = {
                        'bbox': [gt_x1, gt_y1, gt_x2, gt_y2],
                        'segmentation': [instance_polygon.reshape(-1, ).tolist()],
                    }

                    self.all_annotations.append(obj_ann)

        print("{} annotations extracted.".format(len(self.all_annotations)))

    def __len__(self):
        return len(self.all_annotations)

    def __getitem__(self, index):
        ann = self.all_annotations[index]

        # bbox transform.
        bbox = np.array([ann["bbox"]])  # xmin, ymin, xmax, ymax
        bbox = Boxes(bbox)

        # mask transform.
        # print(bbox)
        # print(ann["segmentation"])
        mask = PolygonMasks([ann["segmentation"]])
        mask = mask.crop_and_resize(bbox.tensor, self.size).float()

        return mask

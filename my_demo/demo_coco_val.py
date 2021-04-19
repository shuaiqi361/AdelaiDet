# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
import numpy as np

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"
COLOR_WORLD = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 'orange', 'red', 'maroon',
               'fuchsia', 'purple', 'black', 'gray', 'silver']
RGB_DICT = {'navy': (0, 38, 63), 'blue': (0, 120, 210), 'aqua': (115, 221, 252), 'teal': (15, 205, 202),
            'olive': (52, 153, 114), 'green': (0, 204, 84), 'lime': (1, 255, 127), 'yellow': (255, 216, 70),
            'orange': (255, 125, 57), 'red': (255, 47, 65), 'maroon': (135, 13, 75), 'fuchsia': (246, 0, 184),
            'purple': (179, 17, 193), 'black': (24, 24, 24), 'gray': (168, 168, 168), 'silver': (220, 220, 220)}

for k, v in RGB_DICT.items():
    RGB_DICT[k] = (v[2], v[1], v[0])  # RGB to BGR


def switch_tuple(input_tuple):
    return (input_tuple[2], input_tuple[1], input_tuple[0])


nice_colors = {
    'person': switch_tuple(RGB_DICT['orange']), 'car': switch_tuple(RGB_DICT['green']),
    'bus': switch_tuple(RGB_DICT['lime']), 'truck': switch_tuple(RGB_DICT['olive']),
    'bicycle': switch_tuple(RGB_DICT['maroon']), 'motorcycle': switch_tuple(RGB_DICT['fuchsia']),
    'cyclist': switch_tuple(RGB_DICT['yellow']), 'pedestrian': switch_tuple(RGB_DICT['orange']),
    'tram': switch_tuple(RGB_DICT['purple']), 'van': switch_tuple(RGB_DICT['teal']),
    'misc': switch_tuple(RGB_DICT['navy'])
}


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.SMInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.DTInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--coco-path", type=str,
                        default='/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/coco',
                        help="root_path to the coco dataset")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    # Loading COCO validation images
    args.data_type = 'val2017'
    # annotation_file = '{}/annotations/instances_{}.json'.format(args.coco_path, args.data_type)
    if 'test' not in args.data_type:
        annotation_file = '{}/annotations/instances_{}.json'.format(args.coco_path, args.data_type)
        dataset_name = 'coco_2017_val'
    else:
        annotation_file = '{}/annotations/image_info_test-dev2017.json'.format(args.coco_path)
        dataset_name = 'coco_2017_train'

    coco = COCO(annotation_file)
    imgIds = coco.getImgIds()
    pred_codes = []

    for img_id in imgIds:
        img = coco.loadImgs(img_id)[0]
        image_path = '%s/%s/%s' % (args.coco_path, args.data_type, img['file_name'])
        w_img = int(img['width'])
        h_img = int(img['height'])
        if w_img < 1 or h_img < 1:
            continue

        # original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # use PIL, to be consistent with evaluation
        img = read_image(image_path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        # print(predictions["instances"])
        # codes_ = predictions["instances"].pred_codes.cpu().numpy()
        # pred_codes.append(codes_)

        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                image_path, len(predictions["instances"]), time.time() - start_time
            )
        )

        # # plot histogram of codes
        # fig = plt.figure()
        # arr = plt.hist(codes_.reshape((-1,)).tolist(), bins=30, color='g', density=True)
        # plt.rcParams.update({'font.size': 8})
        # plt.xlabel('Sparse Codes')
        # plt.xlabel('Counts')
        # plt.title('Histogram of sparse codes')
        # plt.show()

        # show ground truth
        ann_ids = coco.getAnnIds(imgIds=img_id)
        gt_anns = coco.loadAnns(ids=ann_ids)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        print('Loading image of id:', img_id)

        # plotting the groundtruth
        gt_image = image.copy()
        gt_blend_mask = np.zeros(shape=gt_image.shape, dtype=np.uint8)
        for ann_ in gt_anns:
            if ann_['iscrowd'] == 1:
                continue
            polygons_ = ann_['segmentation']
            use_color_key = COLOR_WORLD[random.randint(1, len(COLOR_WORLD)) - 1]
            for poly in polygons_:
                poly = np.array(poly).reshape((-1, 2))
                cv2.polylines(gt_image, [poly.astype(np.int32)], True,
                              color=switch_tuple(RGB_DICT[use_color_key]),
                              thickness=2)
                cv2.drawContours(gt_blend_mask, [poly.astype(np.int32)], contourIdx=-1,
                                 color=switch_tuple(RGB_DICT[use_color_key]),
                                 thickness=-1)

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(image_path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        else:
            # continue
            # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])

            gt_dst_img = cv2.addWeighted(gt_image, 0.4, gt_blend_mask, 0.6, 0)
            gt_dst_img[gt_blend_mask == 0] = gt_image[gt_blend_mask == 0]

            cat_image = np.concatenate([visualized_output.get_image()[:, :, ::-1], gt_dst_img], axis=1)

            cv2.imshow('Pred vs. GT', cat_image)

            if cv2.waitKey() & 0xFF == ord('q'):
                break

    # pred_codes = np.concatenate(pred_codes, axis=0)
    # sparsity_counts = np.sum(np.abs(pred_codes) > 1e-2)
    # num_obj, num_dim = pred_codes.shape
    # print('Overall sparsity: ', sparsity_counts * 1. / (num_obj * num_dim))
    #
    # kur = np.sum(kurtosis(pred_codes, axis=1, fisher=True, bias=False)) / num_obj
    # print('Overall Kurtosis: ', kur)

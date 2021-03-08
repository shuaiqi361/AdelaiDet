# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
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
        default=0.5,
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
    args.data_type = 'test2017'
    # annotation_file = '{}/annotations/instances_{}.json'.format(args.coco_path, args.data_type)
    if 'test' not in args.data_type:
        annotation_file = '{}/annotations/instances_{}.json'.format(args.coco_path, args.data_type)
        dataset_name = 'coco_2017_val'
    else:
        annotation_file = '{}/annotations/image_info_test-dev2017.json'.format(args.coco_path)
        dataset_name = 'coco_2017_train'

    coco = COCO(annotation_file)
    imgIds = coco.getImgIds()

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
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                image_path, len(predictions["instances"]), time.time() - start_time
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(image_path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        else:
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            print('image id: ', img_id)
            if cv2.waitKey() & 0xFF == ord('q'):
                break

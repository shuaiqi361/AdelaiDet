# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import multiprocessing as mp
import torch
import time
import cv2
import json
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
import numpy as np

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.evaluation import COCOEvaluator

# from predictor import VisualizationDemo
from detectron2.engine.defaults import DefaultPredictor
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
        "--result-dir",
        type=str,
        default='/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/ablation/3090/Res_50_SMInst_017',
        help="A file or directory to save output coco json. ",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.05,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--mask-size",
        type=int,
        default=28,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    _cpu_device = torch.device("cpu")

    # demo = VisualizationDemo(cfg)
    predictor = DefaultPredictor(cfg)

    # Loading COCO validation images
    dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
    args.data_type = 'val2017'
    if 'test' not in args.data_type:
        annotation_file = '{}/annotations/instances_{}.json'.format(args.coco_path, args.data_type)
        dataset_name = 'coco_2017_train'
    else:
        annotation_file = '{}/annotations/image_info_test-dev2017.json'.format(args.coco_path)
        dataset_name = 'coco_2017_val'

    coco = COCO(annotation_file)
    imgIds = coco.getImgIds()
    seg_results = []

    # load the class conditional masks dictionary, the key is the class name
    save_data_root = os.path.join(dataDir, 'sparse_shape_dict')
    out_class_conditional_masks = '{}/class_conditional_masks_m{}_id_key.json'.format(save_data_root, args.mask_size)

    with open(out_class_conditional_masks, 'r') as f_out:
        class_conditional_masks = json.load(f_out)
        print("Keys: ", class_conditional_masks.keys())

    # unmap the category ids for COCO
    coco_evaluator = COCOEvaluator(dataset_name, cfg, True, output_dir=args.result_dir)
    if hasattr(coco_evaluator._metadata, "thing_dataset_id_to_contiguous_id"):
        dataset_id_to_contiguous_id = coco_evaluator._metadata.thing_dataset_id_to_contiguous_id
        all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
        num_classes = len(all_contiguous_ids)
        assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1
        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}

    for img_id in imgIds:
        img = coco.loadImgs(img_id)[0]
        image_path = '%s/%s/%s' % (args.coco_path, args.data_type, img['file_name'])
        w_img = int(img['width'])
        h_img = int(img['height'])
        if w_img < 1 or h_img < 1:
            continue

        # use PIL, to be consistent with evaluation
        img = read_image(image_path, format="BGR")
        start_time = time.time()
        # predictions, _ = demo.run_on_image(img)
        predictions = predictor(img)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                image_path, len(predictions["instances"]), time.time() - start_time
            )
        )

        if "instances" in predictions:
            instances = predictions["instances"].to(_cpu_device)
            predictions["instances"] = instances_to_coco_json(instances, img_id)
        else:
            raise NotImplementedError

        for result in predictions["instances"]:
            category_id = result["category_id"]
            result["category_id"] = reverse_id_mapping[category_id]
            # print(result["category_id"], " type: ", type(result["category_id"]))
            gt_x1, gt_y1, gt_w, gt_h = result['bbox']
            gt_x1 = int(gt_x1)
            gt_y1 = int(gt_y1)
            gt_w = int(gt_w)
            gt_h = int(gt_h)

            # substitute in the class conditional masks
            recon_contour = np.clip(np.array(class_conditional_masks[str(result["category_id"])]) * 255., 0, 255)
            recon_contour = recon_contour.astype(np.uint8)
            recon_ori_masks = cv2.resize(recon_contour, dsize=(gt_w, gt_h))
            recon_ori_masks = np.where(recon_ori_masks >= 0.5 * 255, 1, 0).astype(np.uint8)

            # convert reconstructed masks to original rle masks
            recon_m = np.zeros((h_img, w_img), dtype=np.uint8)
            recon_m[gt_y1:gt_y1 + gt_h, gt_x1:gt_x1 + gt_w] = recon_ori_masks
            rle_new = encode_mask(recon_m.astype(np.uint8))
            result['segmentation'] = rle_new

            seg_results.append(result)

    with open('{}/results/{}_seg_results.json'.format(args.result_dir, args.data_type), 'w') as f_det:
        json.dump(seg_results, f_det)

    if 'test' not in args.data_type:
        print('---------------------------------------------------------------------------------')
        print('Running COCO segmentation val17 evaluation ...')
        coco_pred = coco.loadRes('{}/results/{}_seg_results.json'.format(args.result_dir, args.data_type))
        coco_eval = COCOeval(coco, coco_pred, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

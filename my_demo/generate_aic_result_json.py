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
nice_colors = {1: (10, 250, 10), 2: (200, 10, 10)}


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
    parser.add_argument("--data-path", type=str,
                        default='/home/keyi/Documents/Data/AIC/Baidu',
                        help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--coco-path", type=str,
                        default='/home/keyi/Documents/Data/COCO_17',
                        help="root_path to the coco dataset")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default='/home/keyi/Documents/research/code/AdelaiDet/experiments/2080_res50_DTInst_002',
        help="A file or directory to save output coco json. ",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
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
    _cpu_device = torch.device("cpu")

    # demo = VisualizationDemo(cfg)
    predictor = DefaultPredictor(cfg)

    # Loading COCO validation images
    args.data_type = 'val'
    annotation_file = '{}/1584811984_train_final.json'.format(args.data_path)
    dataset_name = 'coco_2017_val'

    coco = COCO(annotation_file)
    imgIds = coco.getImgIds()
    seg_results = []
    img_counter = 0
    valid_ids = []

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
        image_path = '%s/train/%s' % (args.data_path, img['file_name'].split('/')[-1])
        if not os.path.exists(image_path):
            continue
        else:
            img_counter += 1
            valid_ids.append(img_id)

        # # plot ground truth cars
        # output_image = cv2.imread(image_path)
        # ann_ids = coco.getAnnIds(imgIds=img_id)
        # gt_anns = coco.loadAnns(ids=ann_ids)

        # for ann_ in gt_anns:
            # x1, y1, w, h = ann_['bbox']
            # bbox = [x1, y1, x1 + w, y1 + h]
            # text = 'class: ' + str(ann_["category_id"])
            # label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            # text_location = [int(bbox[0]) + 1, int(bbox[1]) + 1,
            #                  int(bbox[0]) + 1 + label_size[0][0],
            #                  int(bbox[1]) + 1 + label_size[0][1]]
            # cv2.rectangle(output_image, pt1=(int(bbox[0]), int(bbox[1])),
            #               pt2=(int(bbox[2]), int(bbox[3])),
            #               color=nice_colors[ann_["category_id"]], thickness=2)
            # cv2.putText(output_image, text, org=(int(text_location[0]), int(text_location[3])),
            #             fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.5,
            #             color=nice_colors[ann_["category_id"]])
            # det = {
            #     'image_id': img_id,
            #     'category_id': ann_["category_id"],
            #     'score': 1.0,
            #     'bbox': ann_['bbox']
            # }
            # seg_results.append(det)

        # cv2.imshow('GT bbox', output_image)
        # if cv2.waitKey() & 0xFF == ord('q'):
        #     exit()
        #
        # continue

        # # use PIL, to be consistent with evaluation
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
            _id = reverse_id_mapping[category_id]
            if _id == 3:  # cars
                result["category_id"] = 1
            elif _id == 6:  # buses
                result["category_id"] = 2
            elif _id == 8:  # trucks
                result["category_id"] = 2
            else:
                continue

            seg_results.append(result)

    with open('{}/results/{}_baidu_bbox_results_02.json'.format(args.result_dir, args.data_type), 'w') as f_det:
        json.dump(seg_results, f_det)

    # # with open('{}/1584811984_train_final_small.json'.format(args.data_path), 'w') as f_det:
    # #     json.dump(seg_results, f_det)
    #
    print('Totally, {} images have been evaluated.'.format(img_counter))

    if 'test' not in args.data_type:
        print('---------------------------------------------------------------------------------')
        print('Running bbox evaluation on Baidu finetune data ...')
        coco_pred = coco.loadRes('{}/results/{}_baidu_bbox_results_02.json'.format(args.result_dir, args.data_type))
        coco_eval = COCOeval(coco, coco_pred, 'bbox')
        coco_eval.params.imgIds = valid_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

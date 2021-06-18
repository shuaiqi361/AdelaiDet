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

from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.evaluation import COCOEvaluator

# from predictor import VisualizationDemo
from detectron2.engine.defaults import DefaultPredictor
from adet.config import get_cfg


# constants
RGB_DICT = {'navy': (0, 38, 63), 'blue': (0, 120, 210), 'aqua': (115, 221, 252), 'teal': (15, 205, 202),
            'olive': (52, 153, 114), 'green': (0, 204, 84), 'lime': (1, 255, 127), 'yellow': (255, 216, 70),
            'orange': (255, 125, 57), 'red': (255, 47, 65), 'maroon': (135, 13, 75), 'fuchsia': (246, 0, 184),
            'purple': (179, 17, 193), 'black': (24, 24, 24), 'gray': (168, 168, 168), 'silver': (220, 220, 220)}


def switch_tuple(input_tuple):
    return (input_tuple[2], input_tuple[1], input_tuple[0])


nice_colors = {
    'car': switch_tuple(RGB_DICT['green']),
    'bus': switch_tuple(RGB_DICT['orange']),
    'truck': switch_tuple(RGB_DICT['orange'])
}
WINDOW_NAME = "COCO detections"
cat_name = {0: u'__background__',
            1: u'person',
            2: u'bicycle',
            3: u'car',
            4: u'motorcycle',
            5: u'airplane',
            6: u'bus',
            7: u'train',
            8: u'truck',
            9: u'boat',
            10: u'traffic light',
            11: u'fire hydrant',
            12: u'stop sign',
            13: u'parking meter',
            14: u'bench',
            15: u'bird',
            16: u'cat',
            17: u'dog',
            18: u'horse',
            19: u'sheep',
            20: u'cow',
            21: u'elephant',
            22: u'bear',
            23: u'zebra',
            24: u'giraffe',
            25: u'backpack',
            26: u'umbrella',
            27: u'handbag',
            28: u'tie',
            29: u'suitcase',
            30: u'frisbee',
            31: u'skis',
            32: u'snowboard',
            33: u'sports ball',
            34: u'kite',
            35: u'baseball bat',
            36: u'baseball glove',
            37: u'skateboard',
            38: u'surfboard',
            39: u'tennis racket',
            40: u'bottle',
            41: u'wine glass',
            42: u'cup',
            43: u'fork',
            44: u'knife',
            45: u'spoon',
            46: u'bowl',
            47: u'banana',
            48: u'apple',
            49: u'sandwich',
            50: u'orange',
            51: u'broccoli',
            52: u'carrot',
            53: u'hot dog',
            54: u'pizza',
            55: u'donut',
            56: u'cake',
            57: u'chair',
            58: u'couch',
            59: u'potted plant',
            60: u'bed',
            61: u'dining table',
            62: u'toilet',
            63: u'tv',
            64: u'laptop',
            65: u'mouse',
            66: u'remote',
            67: u'keyboard',
            68: u'cell phone',
            69: u'microwave',
            70: u'oven',
            71: u'toaster',
            72: u'sink',
            73: u'refrigerator',
            74: u'book',
            75: u'clock',
            76: u'vase',
            77: u'scissors',
            78: u'teddy bear',
            79: u'hair drier',
            80: u'toothbrush'}


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
    cfg.MODEL.DTMRInst.INFERENCE_TH_TEST = args.confidence_threshold
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
    parser.add_argument("--img-dir",
                        default='/media/keyi/Data/Research/traffic/data/Hwy7/new/sc2')
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
        default='/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/3090_res101_DTInst_002',
        help="A file or directory to save output coco json. ",
    )
    parser.add_argument(
        "--output-video-file",
        type=str,
        default='Hwy7_sc2_segm_1080.mkv',
        help="A file or directory to save output coco json. ",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=720,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
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

    output_folder = os.path.join(args.result_dir, 'demo')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    video_out = cv2.VideoWriter(os.path.join(output_folder, args.output_video_file),
                                cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), args.fps,
                                (args.video_width, args.video_height))

    # Loading COCO validation images
    args.data_type = 'test2017'
    if 'test' not in args.data_type:
        annotation_file = '{}/annotations/instances_{}.json'.format(args.coco_path, args.data_type)
        dataset_name = 'coco_2017_train'
    else:
        annotation_file = '{}/annotations/image_info_test-dev2017.json'.format(args.coco_path)
        dataset_name = 'coco_2017_val'

    coco = COCO(annotation_file)
    imgIds = coco.getImgIds()
    seg_results = []

    # unmap the category ids for COCO
    coco_evaluator = COCOEvaluator(dataset_name, cfg, True, output_dir=args.result_dir)
    if hasattr(coco_evaluator._metadata, "thing_dataset_id_to_contiguous_id"):
        dataset_id_to_contiguous_id = coco_evaluator._metadata.thing_dataset_id_to_contiguous_id
        all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
        num_classes = len(all_contiguous_ids)
        assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1
        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}

    frame_list = sorted(os.listdir(args.img_dir))
    n_frames = len(frame_list)

    for frame_id in range(n_frames):
        frame_name = frame_list[frame_id]
        image_path = os.path.join(args.img_dir, frame_name)
        # output_image = cv2.imread(image_path)

        # use PIL, to be consistent with evaluation
        img = read_image(image_path, format="BGR")
        img = cv2.resize(img, (args.video_width, args.video_height), interpolation=cv2.INTER_LINEAR)
        output_image = img.copy()
        start_time = time.time()

        predictions = predictor(img)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                image_path, len(predictions["instances"]), time.time() - start_time
            )
        )

        instances = predictions["instances"].to(_cpu_device)
        num_instance = len(instances)

        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()
        # boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        # masks = masks.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        blend_mask = np.zeros(shape=output_image.shape, dtype=np.uint8)

        for i in range(num_instance):
            category_id = classes[i]
            category_id = reverse_id_mapping[category_id]
            if cat_name[category_id] == 'car':
                label_text = 'car'
            elif cat_name[category_id] == 'bus':
                label_text = 'truck'
            elif cat_name[category_id] == 'truck':
                label_text = 'truck'
            else:
                continue

            bbox = boxes[i]
            text = label_text + ' %.2f' % scores[i]
            label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
            # text_location = [int(bbox[0]) + 2, int(bbox[1]) + 2,
            #                  int(bbox[0]) + 2 + label_size[0][0],
            #                  int(bbox[1]) + 2 + label_size[0][1]]
            # cv2.rectangle(output_image, pt1=(int(bbox[0]), int(bbox[1])),
            #               pt2=(int(bbox[2]), int(bbox[3])),
            #               color=nice_colors[label_text], thickness=2)
            # cv2.putText(output_image, text, org=(int(text_location[0]), int(text_location[3])),
            #             fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.3,
            #             color=nice_colors[label_text])

            # show the segmentation masks for vehicles
            mask = masks[i]
            obj_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(obj_contours) > 1:
                obj_contours = sorted(obj_contours, key=cv2.contourArea)

            for contour in obj_contours:
                polygon = contour.reshape((-1, 2))
                cv2.polylines(output_image, [polygon.astype(np.int32)], True, color=nice_colors[label_text],
                              thickness=2)
                cv2.drawContours(blend_mask, [polygon.astype(np.int32)], contourIdx=-1,
                                 color=nice_colors[label_text],
                                 thickness=-1)

        dst_img = cv2.addWeighted(output_image, 0.4, blend_mask, 0.6, 0)
        dst_img[blend_mask == 0] = output_image[blend_mask == 0]
        output_image = dst_img

        cv2.imshow('Frames', output_image)
        video_out.write(output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    exit()

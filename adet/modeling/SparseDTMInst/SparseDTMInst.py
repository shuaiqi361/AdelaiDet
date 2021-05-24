import math
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, IOULoss, NaiveGroupNorm, GCN
from .SparseDTM_output import DTInstOutputs
from .SparseDTMEncode import DistanceTransformEncoding

__all__ = ["DTInst"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@PROPOSAL_GENERATOR_REGISTRY.register()
class DTInst(nn.Module):
    """
    Implement Sparse Mask Encoding method.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.cfg = cfg
        self.in_features = cfg.MODEL.DTInst.IN_FEATURES
        self.fpn_strides = cfg.MODEL.DTInst.FPN_STRIDES
        self.focal_loss_alpha = cfg.MODEL.DTInst.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.DTInst.LOSS_GAMMA
        self.center_sample = cfg.MODEL.DTInst.CENTER_SAMPLE
        self.strides = cfg.MODEL.DTInst.FPN_STRIDES
        self.radius = cfg.MODEL.DTInst.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.DTInst.INFERENCE_TH_TRAIN
        self.pre_nms_thresh_test = cfg.MODEL.DTInst.INFERENCE_TH_TEST
        self.pre_nms_topk_train = cfg.MODEL.DTInst.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test = cfg.MODEL.DTInst.PRE_NMS_TOPK_TEST
        self.nms_thresh = cfg.MODEL.DTInst.NMS_TH
        self.post_nms_topk_train = cfg.MODEL.DTInst.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test = cfg.MODEL.DTInst.POST_NMS_TOPK_TEST
        self.thresh_with_ctr = cfg.MODEL.DTInst.THRESH_WITH_CTR
        self.mask_size = cfg.MODEL.DTInst.MASK_SIZE

        # fmt: on
        self.iou_loss = IOULoss(cfg.MODEL.DTInst.LOC_LOSS_TYPE)
        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.DTInst.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        self.DTInst_head = DTInstHead(cfg, [input_shape[f] for f in self.in_features])

        self.flag_parameters = cfg.MODEL.DTInst.FLAG_PARAMETERS
        self.mask_encoding = DistanceTransformEncoding(cfg)

    def forward(self, images, features, gt_instances):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            pre_nms_thresh = self.pre_nms_thresh_train
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
            if not self.flag_parameters:
                # encoding parameters.
                components_path = self.cfg.MODEL.DTInst.PATH_DICTIONARY
                parameters = np.load(components_path)
                learned_dict = parameters['shape_basis']
                shape_mean = parameters['shape_mean']
                shape_std = parameters['shape_std']
                device = torch.device(self.cfg.MODEL.DEVICE)
                with torch.no_grad():
                    dictionary = nn.Parameter(torch.from_numpy(learned_dict).float().to(device), requires_grad=False)
                    shape_mean = nn.Parameter(torch.from_numpy(shape_mean).float().to(device), requires_grad=False)
                    shape_std = nn.Parameter(torch.from_numpy(shape_std).float().to(device), requires_grad=False)
                    self.mask_encoding.dictionary = dictionary
                    self.mask_encoding.shape_mean = shape_mean
                    self.mask_encoding.shape_std = shape_std

                self.flag_parameters = True
        else:
            pre_nms_thresh = self.pre_nms_thresh_test
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test

        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        # logits_pred, reg_pred, ctrness_pred, dtm_residuals, mask_regression = self.DTInst_head(features,
        #                                                                                        self.mask_encoding)
        logits_pred, reg_pred, ctrness_pred, mask_regression = self.DTInst_head(features)

        outputs = DTInstOutputs(
            images,
            locations,
            logits_pred,
            reg_pred,
            ctrness_pred,
            mask_regression,
            self.mask_encoding,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            self.center_sample,
            self.sizes_of_interest,
            self.strides,
            self.radius,
            self.DTInst_head.num_classes,
            pre_nms_thresh,
            pre_nms_topk,
            self.nms_thresh,
            post_nms_topk,
            self.thresh_with_ctr,
            gt_instances,
            cfg=self.cfg
        )

        if self.training:
            losses, _ = outputs.losses()
            return None, losses
        else:
            proposals = outputs.predict_proposals()
            return proposals, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    @staticmethod
    def compute_locations_per_level(h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class DTInstHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        self.num_classes = cfg.MODEL.DTInst.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.DTInst.FPN_STRIDES
        self.num_codes = cfg.MODEL.DTInst.NUM_CODE
        self.use_gcn_in_mask = cfg.MODEL.DTInst.USE_GCN_IN_MASK
        self.gcn_kernel_size = cfg.MODEL.DTInst.GCN_KERNEL_SIZE
        self.mask_size = cfg.MODEL.DTInst.MASK_SIZE
        self.if_whiten = cfg.MODEL.SMInst.WHITEN

        head_configs = {"cls": (cfg.MODEL.DTInst.NUM_CLS_CONVS,
                                cfg.MODEL.DTInst.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.DTInst.NUM_BOX_CONVS,
                                 cfg.MODEL.DTInst.USE_DEFORMABLE),
                        "share": (cfg.MODEL.DTInst.NUM_SHARE_CONVS,
                                  cfg.MODEL.DTInst.USE_DEFORMABLE),
                        "mask": (cfg.MODEL.DTInst.NUM_MASK_CONVS,
                                 cfg.MODEL.DTInst.USE_DEFORMABLE)}

        self.type_deformable = cfg.MODEL.DTInst.TYPE_DEFORMABLE
        self.last_deformable = cfg.MODEL.DTInst.LAST_DEFORMABLE
        norm = None if cfg.MODEL.DTInst.NORM == "none" else cfg.MODEL.DTInst.NORM

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                # conv type.
                if use_deformable:
                    if self.last_deformable:
                        if i == num_convs - 1:
                            conv_func = DFConv2d
                            type_func = self.type_deformable
                        else:
                            conv_func = nn.Conv2d
                            type_func = "Conv2d"
                    else:
                        conv_func = DFConv2d
                        type_func = self.type_deformable
                else:
                    conv_func = nn.Conv2d
                    type_func = "Conv2d"
                # conv operation.
                if type_func == "DCNv1":
                    tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=False,
                        with_modulated_dcn=False
                    ))
                elif type_func == "DCNv2":
                    tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=False,
                        with_modulated_dcn=True
                    ))
                elif type_func == "Conv2d":
                    tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                    ))
                else:
                    raise NotImplementedError
                # norm.
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                # activation.
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        # self.residual = nn.Sequential(
        #     nn.Conv2d(in_channels * 2 + self.mask_size ** 2, in_channels, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(32, in_channels),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(32, in_channels),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels, self.mask_size ** 2, kernel_size=1, stride=1, padding=0),
        # )
        # self.residual = nn.Sequential(
        #     nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        if self.use_gcn_in_mask:
            self.mask_pred = GCN(in_channels, self.num_codes, k=self.gcn_kernel_size)
        else:
            self.mask_pred = nn.Conv2d(
                in_channels, self.num_codes, kernel_size=3,
                stride=1, padding=1
            )

        if cfg.MODEL.DTInst.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.ctrness,
            self.mask_tower, self.mask_pred, self.residual
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DTInst.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        ctrness = []
        # dtm_residuals = []
        mask_reg = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)

            # logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved DTInst, instead of exp.
            bbox_reg.append(F.relu(reg))

            # Mask Encoding
            mask_tower = self.mask_tower(feature)
            # mask_code_prediction = self.mask_pred(mask_tower)
            # mask_tower_cat = torch.cat([mask_tower, cls_tower, bbox_tower], dim=1)
            # mask_tower_cat = mask_tower + cls_tower + bbox_tower
            # mask_tower_cat = mask_tower + cls_tower
            mask_tower_cat = mask_tower
            residual_mask = self.residual(mask_tower_cat)
            mask_reg.append(self.mask_pred(residual_mask))

            # cls_tower_cat = torch.cat([mask_tower, cls_tower], dim=1)
            logits.append(self.cls_logits(cls_tower))

            # if self.if_whiten:
            #     init_mask = torch.matmul(mask_code_prediction.permute(0, 2, 3, 1),
            #                              mask_encoding.dictionary) * mask_encoding.shape_std.view(1, 1, 1, -1) + \
            #                 mask_encoding.shape_mean.view(1, 1, 1, -1)
            # else:
            #     init_mask = torch.matmul(mask_code_prediction.permute(0, 2, 3, 1),
            #                              mask_encoding.dictionary) + mask_encoding.shape_mean.view(1, 1, 1, -1)
            # residual_features = torch.cat([cls_tower, bbox_tower, init_mask.permute(0, 3, 1, 2)],
            #                               dim=1)
            # residual_mask = self.residual(residual_features)
            # dtm_residuals.append(residual_mask + init_mask.permute(0, 3, 1, 2))

        # return logits, bbox_reg, ctrness, dtm_residuals, mask_reg
        return logits, bbox_reg, ctrness, mask_reg

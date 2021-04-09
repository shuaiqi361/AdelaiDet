from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True
_C.INPUT.CROP.CROP_INSTANCE = True

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
_C.MODEL.FCOS.YIELD_PROPOSAL = False

# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.DLA = CN()
_C.MODEL.DLA.CONV_BODY = "DLA34"
_C.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.DLA.NORM = "FrozenBN"

# ---------------------------------------------------------------------------- #
# BAText Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BATEXT = CN()
_C.MODEL.BATEXT.VOC_SIZE = 96
_C.MODEL.BATEXT.NUM_CHARS = 25
_C.MODEL.BATEXT.POOLER_RESOLUTION = (8, 32)
_C.MODEL.BATEXT.IN_FEATURES = ["p2", "p3", "p4"]
_C.MODEL.BATEXT.POOLER_SCALES = (0.25, 0.125, 0.0625)
_C.MODEL.BATEXT.SAMPLING_RATIO = 1
_C.MODEL.BATEXT.CONV_DIM = 256
_C.MODEL.BATEXT.NUM_CONV = 2
_C.MODEL.BATEXT.RECOGNITION_LOSS = "ctc"
_C.MODEL.BATEXT.RECOGNIZER = "attn"
_C.MODEL.BATEXT.CANONICAL_SIZE = 96  # largest min_size for level 3 (stride=8)

# ---------------------------------------------------------------------------- #
# BlendMask Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BLENDMASK = CN()
_C.MODEL.BLENDMASK.ATTN_SIZE = 14
_C.MODEL.BLENDMASK.TOP_INTERP = "bilinear"
_C.MODEL.BLENDMASK.BOTTOM_RESOLUTION = 56
_C.MODEL.BLENDMASK.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO = 1
_C.MODEL.BLENDMASK.POOLER_SCALES = (0.25,)
_C.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT = 1.0
_C.MODEL.BLENDMASK.VISUALIZE = False

# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3

# ---------------------------------------------------------------------------- #
# MEInst Head
# ---------------------------------------------------------------------------- #
_C.MODEL.MEInst = CN()

# This is the number of foreground classes.
_C.MODEL.MEInst.NUM_CLASSES = 80
_C.MODEL.MEInst.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.MEInst.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.MEInst.PRIOR_PROB = 0.01
_C.MODEL.MEInst.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.MEInst.INFERENCE_TH_TEST = 0.05
_C.MODEL.MEInst.NMS_TH = 0.6
_C.MODEL.MEInst.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.MEInst.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.MEInst.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.MEInst.POST_NMS_TOPK_TEST = 100
_C.MODEL.MEInst.TOP_LEVELS = 2
_C.MODEL.MEInst.NORM = "GN"  # Support GN or none
_C.MODEL.MEInst.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.MEInst.THRESH_WITH_CTR = True

# Focal loss parameters
_C.MODEL.MEInst.LOSS_ALPHA = 0.25
_C.MODEL.MEInst.LOSS_GAMMA = 2.0
_C.MODEL.MEInst.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.MEInst.USE_RELU = True
_C.MODEL.MEInst.USE_DEFORMABLE = False
_C.MODEL.MEInst.LAST_DEFORMABLE = False
_C.MODEL.MEInst.TYPE_DEFORMABLE = "DCNv2"  # or DCNv2.

# the number of convolutions used in the cls and bbox tower
_C.MODEL.MEInst.NUM_CLS_CONVS = 4
_C.MODEL.MEInst.NUM_BOX_CONVS = 4
_C.MODEL.MEInst.NUM_SHARE_CONVS = 0
_C.MODEL.MEInst.CENTER_SAMPLE = True
_C.MODEL.MEInst.POS_RADIUS = 1.5
_C.MODEL.MEInst.LOC_LOSS_TYPE = 'giou'

# ---------------------------------------------------------------------------- #
# Mask Encoding
# ---------------------------------------------------------------------------- #
# Whether to use mask branch.
_C.MODEL.MEInst.MASK_ON = True
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.MEInst.IOU_THRESHOLDS = [0.5]
_C.MODEL.MEInst.IOU_LABELS = [0, 1]
# Whether to use class_agnostic or class_specific.
_C.MODEL.MEInst.AGNOSTIC = True
# Some operations in mask encoding.
_C.MODEL.MEInst.WHITEN = True
_C.MODEL.MEInst.SIGMOID = True

# The number of convolutions used in the mask tower.
_C.MODEL.MEInst.NUM_MASK_CONVS = 4

# The dim of mask before/after mask encoding.
_C.MODEL.MEInst.DIM_MASK = 60
_C.MODEL.MEInst.MASK_SIZE = 28
_C.MODEL.MEInst.MASK_LOSS_WEIGHT = 1.0
# The default path for parameters of mask encoding.
_C.MODEL.MEInst.PATH_COMPONENTS = "datasets/coco/components/" \
                                   "coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz"
# An indicator for encoding parameters loading during training.
_C.MODEL.MEInst.FLAG_PARAMETERS = False
# The loss for mask branch, can be mse now.
_C.MODEL.MEInst.MASK_LOSS_TYPE = ["mse", "cosine", "kl_softmax"]

# Whether to use gcn in mask prediction.
# Large Kernel Matters -- https://arxiv.org/abs/1703.02719
_C.MODEL.MEInst.USE_GCN_IN_MASK = False
_C.MODEL.MEInst.GCN_KERNEL_SIZE = 9
# Whether to compute loss on original mask (binary mask).
_C.MODEL.MEInst.LOSS_ON_MASK = False
_C.MODEL.MEInst.LOSS_ON_CODE = True

# ---------------------------------------------------------------------------- #
# CondInst Options
# ---------------------------------------------------------------------------- #
_C.MODEL.CONDINST = CN()

# the downsampling ratio of the final instance masks to the input image
_C.MODEL.CONDINST.MASK_OUT_STRIDE = 4
_C.MODEL.CONDINST.MAX_PROPOSALS = -1

_C.MODEL.CONDINST.MASK_HEAD = CN()
_C.MODEL.CONDINST.MASK_HEAD.CHANNELS = 8
_C.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 3
_C.MODEL.CONDINST.MASK_HEAD.USE_FP16 = False
_C.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS = False

_C.MODEL.CONDINST.MASK_BRANCH = CN()
_C.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = 8
_C.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.CONDINST.MASK_BRANCH.CHANNELS = 128
_C.MODEL.CONDINST.MASK_BRANCH.NORM = "BN"
_C.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS = 4
_C.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = False

# ---------------------------------------------------------------------------- #
# TOP Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = "conv"
_C.MODEL.TOP_MODULE.DIM = 16

# ---------------------------------------------------------------------------- #
# BiFPN options
# ---------------------------------------------------------------------------- #

_C.MODEL.BiFPN = CN()
# Names of the input feature maps to be used by BiFPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.OUT_CHANNELS = 160
_C.MODEL.BiFPN.NUM_REPEATS = 6

# Options: "" (no norm), "GN"
_C.MODEL.BiFPN.NORM = ""


# ---------------------------------------------------------------------------- #
# SMInst Head Configuration
# ---------------------------------------------------------------------------- #
_C.MODEL.SMInst = CN()

# This is the number of foreground classes.
_C.MODEL.SMInst.NUM_CLASSES = 80
_C.MODEL.SMInst.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.SMInst.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.SMInst.PRIOR_PROB = 0.01
_C.MODEL.SMInst.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.SMInst.INFERENCE_TH_TEST = 0.05
_C.MODEL.SMInst.NMS_TH = 0.6
_C.MODEL.SMInst.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.SMInst.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.SMInst.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.SMInst.POST_NMS_TOPK_TEST = 100
_C.MODEL.SMInst.TOP_LEVELS = 2
_C.MODEL.SMInst.NORM = "GN"  # Support GN or none
_C.MODEL.SMInst.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.SMInst.THRESH_WITH_CTR = True
_C.MODEL.SMInst.THRESH_WITH_ACTIVE = False

# Focal loss parameters
_C.MODEL.SMInst.LOSS_ALPHA = 0.25
_C.MODEL.SMInst.LOSS_GAMMA = 2.0
_C.MODEL.SMInst.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.SMInst.USE_RELU = True
_C.MODEL.SMInst.USE_DEFORMABLE = False
_C.MODEL.SMInst.LAST_DEFORMABLE = False
_C.MODEL.SMInst.TYPE_DEFORMABLE = "DCNv2"  # or DCNv2.

# the number of convolutions used in the cls and bbox tower
_C.MODEL.SMInst.NUM_CLS_CONVS = 4
_C.MODEL.SMInst.NUM_BOX_CONVS = 4
_C.MODEL.SMInst.NUM_SHARE_CONVS = 0
_C.MODEL.SMInst.CENTER_SAMPLE = True
_C.MODEL.SMInst.POS_RADIUS = 1.5
_C.MODEL.SMInst.LOC_LOSS_TYPE = 'giou'

# ---------------------------------------------------------------------------- #
# Sparse Shape Encoding for Instance Segmentation
# ---------------------------------------------------------------------------- #
# Whether to use mask branch.
_C.MODEL.SMInst.MASK_ON = True
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.SMInst.IOU_THRESHOLDS = [0.5]
_C.MODEL.SMInst.IOU_LABELS = [0, 1]
# Whether to use class_agnostic or class_specific.
_C.MODEL.SMInst.AGNOSTIC = True
# Some operations in mask encoding.
_C.MODEL.SMInst.WHITEN = True
_C.MODEL.SMInst.SIGMOID = True

# The number of convolutions used in the mask tower.
_C.MODEL.SMInst.NUM_MASK_CONVS = 4

# The dim of mask before/after mask encoding.
_C.MODEL.SMInst.DIM_MASK = 60
_C.MODEL.SMInst.MASK_SIZE = 28
_C.MODEL.SMInst.MASK_SPARSE_ALPHA = 0.1
_C.MODEL.SMInst.MASK_SPARSE_WEIGHT = 0.0
_C.MODEL.SMInst.MASK_LOSS_WEIGHT = 1.0
_C.MODEL.SMInst.SPARSITY_LOSS_TYPE = 'L1'
_C.MODEL.SMInst.SPARSITY_KL_RHO = 0.5

# The dim for sparse shape encoding
_C.MODEL.SMInst.NUM_VERTEX = 180
_C.MODEL.SMInst.NUM_CODE = 256
_C.MODEL.SMInst.POLYGON_SPARSE_ALPHA = 0.1
_C.MODEL.SMInst.MAX_ISTA_ITER = 70
# The default path for parameters of mask encoding.
_C.MODEL.SMInst.PATH_DICTIONARY = "/media/keyi/Data/Research/traffic/detection/AdelaiDet/adet/modeling/SMInst/" \
                                  "dictionary/mask_fromMask_basis_m28_n256_a0.10.npy"
# An indicator for encoding parameters loading during training.
_C.MODEL.SMInst.FLAG_PARAMETERS = False
# The loss for mask branch, can be mse now.
_C.MODEL.SMInst.MASK_LOSS_TYPE = ["mse", "cosine", "kl_sigmoid", "kl_softmax"]
_C.MODEL.SMInst.SHAPE_LOSS_TYPE = "piou"

# Whether to use gcn in mask prediction.
# Large Kernel Matters -- https://arxiv.org/abs/1703.02719
_C.MODEL.SMInst.USE_GCN_IN_MASK = False
_C.MODEL.SMInst.GCN_KERNEL_SIZE = 9
# Whether to compute loss on original mask (binary mask).
_C.MODEL.SMInst.LOSS_ON_MASK = False
_C.MODEL.SMInst.LOSS_ON_CODE = True


# ---------------------------------------------------------------------------- #
# DTInst Head Configuration
# ---------------------------------------------------------------------------- #
_C.MODEL.DTInst = CN()

# This is the number of foreground classes.
_C.MODEL.DTInst.NUM_CLASSES = 80
_C.MODEL.DTInst.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.DTInst.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.DTInst.PRIOR_PROB = 0.01
_C.MODEL.DTInst.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.DTInst.INFERENCE_TH_TEST = 0.05
_C.MODEL.DTInst.NMS_TH = 0.6
_C.MODEL.DTInst.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.DTInst.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.DTInst.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.DTInst.POST_NMS_TOPK_TEST = 100
_C.MODEL.DTInst.TOP_LEVELS = 2
_C.MODEL.DTInst.NORM = "GN"  # Support GN or none
_C.MODEL.DTInst.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.DTInst.THRESH_WITH_CTR = True

# Focal loss parameters
_C.MODEL.DTInst.LOSS_ALPHA = 0.25
_C.MODEL.DTInst.LOSS_GAMMA = 2.0
_C.MODEL.DTInst.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.DTInst.USE_RELU = True
_C.MODEL.DTInst.USE_DEFORMABLE = False
_C.MODEL.DTInst.LAST_DEFORMABLE = False
_C.MODEL.DTInst.TYPE_DEFORMABLE = "DCNv2"  # or DCNv2.

# the number of convolutions used in the cls and bbox tower
_C.MODEL.DTInst.NUM_CLS_CONVS = 4
_C.MODEL.DTInst.NUM_BOX_CONVS = 4
_C.MODEL.DTInst.NUM_SHARE_CONVS = 0
_C.MODEL.DTInst.CENTER_SAMPLE = True
_C.MODEL.DTInst.POS_RADIUS = 1.5
_C.MODEL.DTInst.LOC_LOSS_TYPE = 'giou'

# ---------------------------------------------------------------------------- #
# Sparse Encoding for Distance Transformation Map Instance Segmentation
# ---------------------------------------------------------------------------- #
# Whether to use mask branch.
_C.MODEL.DTInst.MASK_ON = True
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.DTInst.IOU_THRESHOLDS = [0.5]
_C.MODEL.DTInst.IOU_LABELS = [0, 1]
# Whether to use class_agnostic or class_specific.
_C.MODEL.DTInst.AGNOSTIC = True
# Some operations in mask encoding.
_C.MODEL.DTInst.WHITEN = True
_C.MODEL.DTInst.SIGMOID = True

# The number of convolutions used in the mask tower.
_C.MODEL.DTInst.NUM_MASK_CONVS = 4

# The dim of mask before/after mask encoding.
_C.MODEL.DTInst.MASK_SIZE = 28
_C.MODEL.DTInst.FOREGROUND_WEIGHTING = 1.0
_C.MODEL.DTInst.BACKGROUND_WEIGHTING = 1.0
_C.MODEL.DTInst.MASK_BIAS = -0.1
_C.MODEL.DTInst.NUM_CODE = 256
_C.MODEL.DTInst.MASK_SPARSE_ALPHA = 0.1
_C.MODEL.DTInst.MASK_SPARSE_WEIGHT = 0.0
_C.MODEL.DTInst.MASK_LOSS_WEIGHT = 1.0
_C.MODEL.DTInst.SPARSITY_LOSS_TYPE = 'L1'

# The dim for sparse shape encoding
_C.MODEL.DTInst.NUM_VERTEX = 180
_C.MODEL.DTInst.POLYGON_SPARSE_ALPHA = 0.30
_C.MODEL.DTInst.MAX_ISTA_ITER = 80
_C.MODEL.DTInst.DIST_TYPE = "L2"
# The default path for parameters of mask encoding.
_C.MODEL.DTInst.PATH_DICTIONARY = "datasets/coco/components/" \
                                   "coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz"
# An indicator for encoding parameters loading during training.
_C.MODEL.DTInst.FLAG_PARAMETERS = False
# The loss for mask branch, can be mse now.
_C.MODEL.DTInst.MASK_LOSS_TYPE = ["mse", "cosine", "kl_softmax", "mask_mse", "hd"]
_C.MODEL.DTInst.SHAPE_LOSS_TYPE = "piou"

# Whether to use gcn in mask prediction.
# Large Kernel Matters -- https://arxiv.org/abs/1703.02719
_C.MODEL.DTInst.USE_GCN_IN_MASK = False
_C.MODEL.DTInst.GCN_KERNEL_SIZE = 9
# Whether to compute loss on original mask (binary mask).
_C.MODEL.DTInst.LOSS_ON_MASK = False
_C.MODEL.DTInst.LOSS_ON_CODE = True

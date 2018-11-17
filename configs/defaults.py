from yacs.config import CfgNode as CN


_C = CN()


"""
MODEL PARAMETERS
"""
_C.MODEL = CN()

_C.MODEL.IMAGE_SIZE = 300
_C.MODEL.FEATURE_MAPS_DIM = [38, 19, 10, 5, 3, 1]
_C.MODEL.FEATURE_MAPS_STRIDES = [8, 16, 32, 64, 100, 300]

_C.MODEL.ANCHOR_MIN_SIZES = [60, 99, 120, 150, 190, 220]
_C.MODEL.ANCHOR_MAX_SIZES = [85, 110, 140, 180, 210, 250]
_C.MODEL.ANCHOR_ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
_C.MODEL.ANCHOR_VARIANCES = [0.1, 0.2]


"""
TRAINING META-PARAMETERS
"""
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.LR = 1e-3
_C.TRAIN.WEIGHT_DECAY = 0.
_C.TRAIN.SCHEDULER_STEP_SIZE = 1000
_C.TRAIN.SCHEDULER_GAMMA = 0.99

_C.TRAIN.NUM_EPOCHS = 50
_C.TRAIN.PAIRS_PER_VIDEO = 1
_C.TRAIN.RESUME_CHECKPOINT = ""

_C.TRAIN.LAMBDA = 1.0  # Loss = loss_classification + lambda * loss_regression
_C.TRAIN.NEGPOS_RATIO = 3  # Negative/positive ratio during training

_C.TRAIN.REGRESSION_LOSS = "smooth_l1"  # Smooth L1 or L1 loss.

# Positive/negative examples sampling
_C.TRAIN.TH_HIGH = 0.6
_C.TRAIN.TH_LOW = 0.3

# Debugging
_C.TRAIN.DEBUG_SEQ = -1

# Data augmentation
# _C.TRAIN.DATA_AUG_TRANSLATION_RANGE = 0.2
# _C.TRAIN.DATA_AUG_RESIZE_SCALE_MIN = 0.08
# _C.TRAIN.DATA_AUG_RESIZE_SCALE_MAX = 1.15


"""
PATHS
"""
_C.PATH = CN()

# TrackingNet dataset root path
_C.PATH.TRACKINGNET = "/cvlabdata1/cvlab/datasets_carlier/TrackingNet"
# OTB dataset root path
_C.PATH.OTB = "/cvlabdata1/cvlab/datasets_hugonot/DATA/OTB"
# ILSVRC dataset root path
_C.PATH.ILSVRC = "/cvlabdata1/cvlab/datasets_hugonot/DATA/ILSVRC"

# COCO Detection 2014 root path
_C.PATH.COCO = "/cvlabdata1/cvlab/datasets_timur/COCO/train2014"
# COCO annotation JSON file path
_C.PATH.COCO_ANN_FILE = "/cvlabdata1/cvlab/datasets_carlier/COCO_annotations/instances_train2014.json"

# Where to save checkpoints, tensorboard runs...
_C.PATH.DATA_DIR = "/cvlabdata2/home/acarlier/project2"


"""
DEBUG
"""
_C.DEBUG = False


# Exporting as cfg is a nice convention
cfg = _C

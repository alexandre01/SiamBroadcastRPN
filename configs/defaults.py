from yacs.config import CfgNode as CN


_C = CN()


"""
MODEL PARAMETERS
"""
_C.MODEL = CN()

_C.MODEL.NET = "Net"

_C.MODEL.Z_SIZE = 300
_C.MODEL.X_SIZE = 300
_C.MODEL.FEATURE_MAPS_DIM = [38, 19, 10, 5, 3, 1]
_C.MODEL.FEATURE_MAPS_STRIDES = [8, 16, 32, 64, 100, 300]

_C.MODEL.ANCHOR_MIN_SIZES = [60, 99, 120, 150, 190, 220]
_C.MODEL.ANCHOR_MAX_SIZES = [85, 110, 140, 180, 210, 250]
_C.MODEL.ANCHOR_ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
_C.MODEL.ANCHOR_VARIANCES = [0.1, 0.2]

# Input tensor should be values between 0 and 1.0
_C.MODEL.INPUT_RANGE = 1.0


"""
TRACKING PARAMETERS
"""
_C.TRACKING = CN()
_C.TRACKING.USE_ENGINEERING = True
_C.TRACKING.LR = 0.295
_C.TRACKING.PENALTY_K = 0.055
_C.TRACKING.WINDOW_INFLUENCE = 0.42
_C.TRACKING.UPDATE_RATE = 0.0

_C.TRACKING.USE_CORRELATION_GUIDE = False

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

_C.TRAIN.USE_MASK = True

# Cropping
_C.TRAIN.CROP_CONTEXT_AMOUNT_Z = 1.0
_C.TRAIN.CROP_CONTEXT_AMOUNT_X = 1.0

# Data augmentation
_C.TRAIN.DATA_AUG_TRANSLATE_RANGE = 0.3
_C.TRAIN.DATA_AUG_RESIZE_SCALE_MIN = 0.35
_C.TRAIN.DATA_AUG_RESIZE_SCALE_MAX = 1.5

_C.TRAIN.FRAME_RANGE = 100

"""
PATHS
"""
_C.PATH = CN()

# TrackingNet dataset root path
_C.PATH.TRACKINGNET = "/PATH/TO/TRACKINGNET"

# UAV dataset root path
_C.PATH.UAV = "/PATH/TO/UAV"

# OTB dataset root path
_C.PATH.OTB = "/PATH/TO/OTB"

# ILSVRC dataset root path
_C.PATH.ILSVRC = "/PATH/TO/ILSVRC"

# COCO Detection 2014 root path
_C.PATH.COCO = "/PATH/TO/COCO"

# COCO annotation JSON file path
_C.PATH.COCO_ANN_FILE = "/PATH/TO/COCO_ANNOTATION_FILE"

# Where to save checkpoints, tensorboard runs...
_C.PATH.DATA_DIR = "/PATH/TO/PROJECT_DIRECTORY"

# Pretrained models
_C.PATH.PRETRAINED_SIAMRPN = "/PATH/TO/pretrained/SiamRPNBIG.model"
_C.PATH.PRETRAINED_SIAMFC = "/PATH/TO/pretrained/siamfc"

# AlexnetBIG weights
_C.PATH.ALEXNETBIG_WEIGHTS = "/PATH/TO/pretrained/alexnetBIG.pth"

"""
DEBUG
"""
_C.DEBUG = False


# Exporting as cfg is a nice convention
cfg = _C

from .siamRPNBIG import SiamRPNBIG
from .siamConcatRPN import SiamConcatRPN
from .siamBroadcastRPN import SiamBroadcastRPN


def load_net(model_name, cfg):
    try:
        return globals()[model_name](cfg)
    except Exception:
        raise Exception("No model named {}".format(model_name))

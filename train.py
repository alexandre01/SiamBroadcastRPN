import argparse
import models
import trainers
from configs import cfg


parser = argparse.ArgumentParser(description="Reference Guided RPN Training")
parser.add_argument(
    "--config-file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

args = parser.parse_args()

if args.config_file:
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

net = models.Net()
trainer = trainers.Trainer(net, cfg)
trainer.train()

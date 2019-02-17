import argparse
import models
import trackers
import experiments
from configs import cfg

parser = argparse.ArgumentParser(description='Benchmark SiamBroadcastRPN on a dataset.')
parser.add_argument("--checkpoint")
parser.add_argument("--visualize", type=bool, default=False)
parser.add_argument("--sequences", nargs='+', default=[])
parser.add_argument("--version", default=2015)
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

if len(args.sequences) == 0:
    args.sequences = None

net = models.load_net(cfg.MODEL.NET, cfg)
tracker = trackers.load_tracker(net, args.checkpoint, cfg)

experiment = experiments.ExperimentOTB(cfg, version=args.version, sequences=args.sequences)

experiment.run(tracker, visualize=args.visualize)
experiment.report([tracker.name], args=args)

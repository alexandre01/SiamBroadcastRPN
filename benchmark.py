import argparse
import models
import trackers
import experiments
from configs import cfg

parser = argparse.ArgumentParser(description='Benchmark model on a dataset.')
parser.add_argument("--checkpoint")
parser.add_argument("--visualize", type=bool, default=False)
parser.add_argument("--version", default=2015)

args = parser.parse_args()

net = models.Net()
tracker = trackers.TrackerGuided(net, args.checkpoint, cfg)
experiment = experiments.ExperimentOTB(cfg, version=args.version)

experiment.run(tracker, visualize=args.visualize)
experiment.report([tracker.name], args=args)

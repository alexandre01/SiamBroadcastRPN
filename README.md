# Reference Guided RPN
Semester Project

Inspired by "Fast Video Object Segmentation by Reference-Guided Mask Propagation" [[paper](http://www.eecs.harvard.edu/~kalyans/research/videosegmentation/FastVideoSegmentation_CVPR18.pdf)] [[code](https://github.com/seoungwugoh/RGMP)].

We use Resnet50 as backbone.

Similarly to SSD, bounding boxes are regressed at different feature maps.

## Training
To start training the network, run the command `python3 train.py --config-file PATH/TO/CONFIG/FIlE`.

## Config files
Configuration files are written in YAML (Yet Another Markup Language), which override the default configuration in `configs/defaults.py`.

# Benchmarking
To be completed.
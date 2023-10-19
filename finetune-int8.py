import numpy as np
from edgeyolo import launch_finetune
import argparse


def make_parser():
    parser = argparse.ArgumentParser("EdgeYOLO train parser")
    parser.add_argument("-c", "--cfg", type=str, default="params/train/train_coco.yaml")

    # Not commend
    parser.add_argument("--default", action="store_true", help="use default train settings in edgeyolo/train/default.yaml")
    return parser.parse_args()


if __name__ == '__main__':
    args = make_parser()
    launch_finetune("DEFAULT" if args.default else args.cfg)

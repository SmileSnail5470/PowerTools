from evaluator import Evaluator
from rgvi import RGVI
from argparse import ArgumentParser
import os
import torch
import warnings
warnings.filterwarnings('ignore')


parser = ArgumentParser()
parser.add_argument('--root', default='input/', type=str, help='root directory of videos')
parser.add_argument('--res', default='480p', choices=['240p', '480p', '2K'], help='input resolution')
parser.add_argument('--prompt', default=None, type=str, help='text prompt for generative model')
args = parser.parse_args()


if __name__ == '__main__':

    # define model
    model = RGVI().eval()

    # testing stage
    with torch.no_grad():
        evaluator = Evaluator(args.root, args.res)
        evaluator.evaluate(model, args.prompt, "output")

"""
train code of DARTS
"""
import sys
import logging
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from DARTS import genotypes
from DARTS.model import NetworkCIFAR as Network

CLASSES = 10  # # of classes


def main():
    """
    main function
    """
    if not torch.cuda.is_available():
        logger.info("No GPU device available")
        sys.exit(1)
    logger.info("GPU device available")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = (
        True  # benchmark mode to find the best algorithm unless the input size doesn't vary
    )
    logger.info("gpu device = %d", args.gpu)
    logger.info("args = %s", args)

    if args.DARTS:
        genotype = genotypes.DARTS
    else:
        genotype = genotypes.PRIMITIVES
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    # TODO: Code below


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")

    parser.add_argument(
        "--DARTS", type=str, action="store_true", help="use DARTS architecture or not"
    )
    parser.add_argument("--init_channels", type=int, default=36, help="num of init channels")
    parser.add_argument("--layers", type=int, default=20, help="total number of layers")
    parser.add_argument("--auxiliary", action="store_true", help="use auxiliary tower")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    main()

import sys
import logging
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn

def main():
    if not torch.cuda.is_available():
        logger.info('No GPU device available')
        sys.exit(1)
    logger.info('GPU device available')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    main()
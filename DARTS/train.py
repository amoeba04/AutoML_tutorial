import torch
import sys
import logging

def main():
    if not torch.cuda.is_available():
        logger.info('No GPU device available')
        sys.exit(1)
    logger.info('GPU device available')


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    main()
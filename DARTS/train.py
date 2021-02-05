"""
train code of DARTS
"""
import os
import sys
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
from torch.autograd import Variable

from DARTS import genotypes
from DARTS import utils
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
        # TODO: fix genotypes.*
        genotype = genotypes.DARTS
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logger.info("Param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2
    )
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    for epoch in range(args.epochs):
        logger.info("epoch %d lr %e", epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logger.info("train_acc %f", train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logger.info("valid_acc %f", valid_acc)

        scheduler.step()

        utils.save(model, os.path.join(args.save, "weights.pt"))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss, n)
        top1.update(prec1, n)
        top5.update(prec5, n)

        if step % args.report_freq == 0:
            logger.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss, n)
        top1.update(prec1, n)
        top5.update(prec5, n)

        if step % args.report_freq == 0:
            logger.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--data", type=str, default="../data", help="location of the data corpus")
    parser.add_argument("--save", type=str, default="EXP", help="experiment name")
    parser.add_argument("--epochs", type=int, default=1, help="num of training epochs")
    parser.add_argument("--report_freq", type=float, default=50, help="report frequency")

    parser.add_argument("--DARTS", action="store_true", help="use DARTS architecture or not")
    parser.add_argument("--init_channels", type=int, default=36, help="num of init channels")
    parser.add_argument("--layers", type=int, default=20, help="total number of layers")
    parser.add_argument("--auxiliary", action="store_true", help="use auxiliary tower")
    parser.add_argument(
        "--auxiliary_weight", type=float, default=0.4, help="weight for auxiliary loss"
    )
    parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--drop_path_prob", type=float, default=0.2, help="drop path probability")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    main()

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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
from torch.autograd import Variable

from DARTS import utils
from DARTS.model_search import Network
from DARTS.architect import Architect

CLASSES = 10  # # of classes


def main():
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

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model = model.cuda()
    logger.info("Param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=2,
    )
    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
        num_workers=2,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )

    architect = Architect(model, args)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        logger.info("epoch %d lr %e", epoch, lr)

        genotype = model.genotype()
        logger.info("genotype = %s", genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        train_acc, train_obj = train(
            train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch
        )
        logger.info("train_acc %f", train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logger.info("valid_acc %f", valid_acc)

        scheduler.step()

        utils.save(model, os.path.join(args.save, "weights.pt"))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()

        # get a random minibatch from the train queue with replacement
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

        architect.step(
            input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled
        )

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logger.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

        if step % args.save_freq == 0:
            utils.save(
                model, os.path.join(args.save, "weights_" + str(epoch) + "_" + str(step) + ".pt")
            )

    return top1.avg, objs.avg


@torch.no_grad()
def infer(valid_queue, model, criterion):
    # TODO: fix memory overusing
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logger.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    # parser.add_argument("--data", type=str, default="../data", help="location of the data corpus")
    # parser.add_argument(
    #     "--save", type=str, default="./DARTS/checkpoints", help="location of the checkpoints"
    # )
    # parser.add_argument("--epochs", type=int, default=1, help="num of training epochs")
    # parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
    # parser.add_argument("--save_freq", type=float, default=500, help="save frequency")

    # parser.add_argument("--DARTS", action="store_true", help="use DARTS architecture or not")
    parser.add_argument("--init_channels", type=int, default=36, help="num of init channels")
    parser.add_argument("--layers", type=int, default=20, help="total number of layers")
    # parser.add_argument("--auxiliary", action="store_true", help="use auxiliary tower")
    # parser.add_argument(
    #     "--auxiliary_weight", type=float, default=0.4, help="weight for auxiliary loss"
    # )
    parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate")
    parser.add_argument("--learning_rate_min", type=float, default=0.001, help="min learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
    parser.add_argument("--train_portion", type=float, default=0.5, help="portion of training data")
    # parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    # parser.add_argument("--drop_path_prob", type=float, default=0.2, help="drop path probability")
    # parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    # parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
    # parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
    parser.add_argument(
        "--arch_learning_rate", type=float, default=3e-4, help="learning rate for arch encoding"
    )
    parser.add_argument(
        "--arch_weight_decay", type=float, default=1e-3, help="weight decay for arch encoding"
    )
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    main()
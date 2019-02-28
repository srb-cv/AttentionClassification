import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict

from spatial_attention_model import AttnVGG_spatial
from model1 import AttnVGG_before
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loaddata import LoadData
from Train_VGG import vgg_512fc
from datasets.transformation import augmentation,conversion
from datasets import Tobacco
from sklearn.ensemble import VotingClassifier


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--num_classes',default=10, type=int, help='num of class in the model')
parser.add_argument('--train_attn', default='train_attn', action='store_true',
                    help='Train the model with Attn')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def load_weight(model, path):
    state_dict = torch.load(path)
    new_dict = OrderedDict
    # print(state_dict['state_dict'])
    # for k, v,_ in state_dict['state_dict']:
    #     name = k[7:]
    #     new_dict[name] = v
    new_dict = {str.replace(k, 'module.', ''): v for k, v in state_dict[
        'state_dict'].items()}
    model.load_state_dict(new_dict)
    return model


class MyEnsemble(nn.Module):
    #def __init__(self, model_header, model_footer):
    def __init__(self, model_header, model_footer, model_left, model_right):
        super(MyEnsemble, self).__init__()
        self.modelA = model_header
        self.modelB = model_footer
        self.modelC = model_left
        self.modelD = model_right

        for param in self.parameters():
            param.requires_grad = False

        #self.classifier = nn.Linear(10*2, 10)
        self.classifier = nn.Linear(10 * 4, 10)

    #def forward(self, x1, x2):
    def forward(self, x1, x2, x3, x4):
        x1, _,  _, _ = self.modelA(x1)
        x2, _,  _, _ = self.modelB(x2)
        x3, _, _, _ = self.modelC(x3)
        x4, _, _, _ = self.modelD(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        #x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model_header = AttnVGG_spatial(num_classes=10)
    model_footer = AttnVGG_spatial(num_classes=10)
    model_left = AttnVGG_spatial(num_classes=10)
    model_right = AttnVGG_spatial(num_classes=10)
    model_header = load_weight(model_header, "zoo/Header_acc_88_att_tobacco_cropped/model_best.pth.tar")
    model_footer = load_weight(model_footer, "zoo/Footer_acc_88_att_tobacco_cropped/model_best.pth.tar")
    model_left = load_weight(model_left, "zoo/Left_acc_88_att_tobacco_cropped/model_best.pth.tar")
    model_right = load_weight(model_right, "zoo/Footer_acc_88_att_tobacco_cropped/model_best.pth.tar")
    model = MyEnsemble(model_header, model_footer,model_left,model_right)
    #model = MyEnsemble(model_header, model_footer)
    print(model)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    #target_width, target_height = 240, 320
    target_width, target_height = 224, 224

    preprocess_imgs = [
        augmentation.DownScale(target_resolution=(target_width, target_height)),
        conversion.ToFloat(),
        conversion.TransposeImage(),
        conversion.ToTensor()
    ]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    footer_path = '/scratch/Segregated_backup/Footer'
    header_path = '/scratch/Segregated_backup/Header'
    left_path = '/scratch/Segregated_backup/Left'
    right_path = '/scratch/Segregated_backup/Right'

    train_footer_path = os.path.join(footer_path, 'train')
    train_header_path = os.path.join(header_path, 'train')
    train_left_path = os.path.join(left_path, 'train')
    train_right_path = os.path.join(right_path, 'train')

    train_dataset = LoadData(train_header_path, train_footer_path, train_left_path, train_right_path, 3, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, sampler=None)

    val_footer_path = os.path.join(footer_path, 'val')
    val_header_path = os.path.join(header_path, 'val')
    val_left_path = os.path.join(left_path, 'val')
    val_right_path = os.path.join(right_path, 'val')

    val_dataset = LoadData(val_header_path, val_footer_path, val_left_path, val_right_path, 3, transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # val_footer_path = os.path.join(footer_path, 'val')
    # val_header_path = os.path.join(header_path, 'val')
    # val_left_path = os.path.join(left_path, 'val')
    # val_right_path = os.path.join(right_path, 'val')
    #
    # val_dataset = LoadData(val_header_path, val_footer_path, val_left_path, val_right_path, 3, transform)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    model.cuda()
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

    # evaluate on test set
    #test_acc = test(test_loader, model, criterion, args)
    # print("Average test accuracy",test_acc)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (header,footer,left,right, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        use_cuda = torch.cuda.is_available()
        header = header.cuda(args.gpu, non_blocking=True)
        footer = footer.cuda(args.gpu, non_blocking=True)
        left = left.cuda(args.gpu, non_blocking=True)
        right = right.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        #output = model(header,footer)
        output = model(header, footer, left, right)
        loss = criterion(output, target.squeeze())
        # pred = torch.argmax(output, dim=1)
        # correct = pred.eq(target.data).cpu().sum()
        # print("accuracy: %.2f" %(correct/target.size(0)))
        # measure accuracy and record loss

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), header.size(0))
        top1.update(acc1[0], header.size(0))
        top5.update(acc5[0], header.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (header,footer,left,right, target) in enumerate(val_loader):
            header = header.cuda(args.gpu, non_blocking=True)
            footer = footer.cuda(args.gpu, non_blocking=True)
            left = left.cuda(args.gpu, non_blocking=True)
            right = right.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.train_attn:
                #output = model(header, footer)
                output = model(header, footer, left, right)
            else:
                #output = model(header, footer)
                output = model(header, footer, left, right)
            loss = criterion(output, target.squeeze())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), header.size(0))
            top1.update(acc1[0], header.size(0))
            top5.update(acc5[0], header.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

#
# def test(test_loader, model, criterion, args):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         end = time.time()
#         for i, (header, footer, left, right, target) in enumerate(val_loader):
#             header = header.cuda(args.gpu, non_blocking=True)
#             footer = footer.cuda(args.gpu, non_blocking=True)
#             target = target.cuda(args.gpu, non_blocking=True)
#
#             # compute output
#             if args.train_attn:
#                 output = model(header, footer)
#             else:
#                 output = model(header, footer)
#             loss = criterion(output, target.squeeze())
#
#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), header.size(0))
#             top1.update(acc1[0], header.size(0))
#             top5.update(acc5[0], header.size(0))
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             if i % args.print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     i, len(val_loader), batch_time=batch_time, loss=losses,
#                     top1=top1, top5=top5))
#
#         print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#               .format(top1=top1, top5=top5))
#
#     return top1.avg
#


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
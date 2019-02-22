from rvlcdip.data import LoadData
from rvlcdip.model1 import AttnVGG_before

import argparse
from matplotlib import pyplot as plt
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.autograd import Variable
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
from rvlcdip.inference import test


def plot(losses, args):
    plt.plot(losses, label='loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)
    plt.savefig(os.path.join(args.plot_path, 'loss.png'))


def main(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    load_train_data = LoadData(args.dataset_path, 'train', transforms_train)
    train_loader = DataLoader(load_train_data, shuffle=True, num_workers=8, batch_size=args.batch_size)

    transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    val_data = LoadData(args.dataset_path, 'val', transforms_val)
    val_loader = DataLoader(val_data, batch_size=2, num_workers=8)

    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    test_data = LoadData(args.dataset_path, 'test', transforms_test)
    test_loader = DataLoader(test_data, batch_size=2, num_workers=8)

    model = AttnVGG_before(16)
    '''
    model = vgg16(pretrained=False).cuda()
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 16)])
    model.classifier = nn.Sequential(*features)
    print(model)
    '''
    model.cuda()

    optimizer = SGD(model.parameters(), 1e-2,
                                momentum=0.9, weight_decay=5e-4)
    losses = []
    criterion = nn.CrossEntropyLoss()

    if not os.path.exists(args.logs):
        os.makedirs(args.logs)

    writer = SummaryWriter(args.logs)
    running_avg_accuracy = 0
    step = 0
    for epoch in range(args.nepoch):
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        for i, (im, label) in enumerate(train_loader):
            model.train()
            im = im.float()
            im = Variable(im.cuda())
            label = label.cuda()
            optimizer.zero_grad()
            output, _, _, _ = model(im)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            if i == 0:
               losses.append(loss)
            if i % 10 == 0:
                pred = torch.argmax(output, 1)
                total = label.size(0)
                correct = torch.eq(pred, label).sum().double().item()
                accuracy = correct / total
                print("epoch/iteration: %d/%d, loss: %.4f, accuracy: %.4f" %(epoch, i, loss, accuracy))

                running_avg_accuracy = 0.9 * running_avg_accuracy + 0.1 * accuracy
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/accuracy', accuracy, step)
                writer.add_scalar('train/running_avg_accuracy', running_avg_accuracy, step)
            if i > 30:
                break

            step += 1
        test(model, val_loader, writer, epoch=epoch)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    model_name = os.path.join(args.model_path)
    torch.save(model.state_dict(), model_name)

    test(model, test_loader, writer, vis=True)
    plot(losses, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for VGG16 with Attention')
    parser.add_argument('--dataset_path', default='/scratch/Datasets/rvl-cdip', help='provide path to the dataset')
    parser.add_argument('--nepoch', default=5, help='number of epochs')
    parser.add_argument('--plot_path', default='plot', help='provide path for saving loss plot')
    parser.add_argument('--model_path', default='models', help='provide path for saving model')
    parser.add_argument('--batch_size', default=6, help='provide batch size for training')
    parser.add_argument('--logs', default='logs', help='provide path for saving logs')
    args = parser.parse_args()
    main(args)
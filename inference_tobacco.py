import argparse
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torchvision import utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.datasets as datasets

from utilities import visualize_attn_softmax
from model1 import AttnVGG_before


def test(model, test_loader, writer, epoch=0, vis=False):
    correct = 0
    total = 0

    count = 0
    if vis :
        images_dis = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.cuda()
            print(len(images))
            labels = labels.cuda()
            outputs, c1, c2, c3 = model(images)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().double().item()
            if count > 10:
                break
            if vis and count == 0:
                images_dis.append(images[0:36])
            count += 1
    print('Epoch: %d. Test Accuracy of the model on the %d test images: %.3f'%(epoch, 10, 100 * correct / total))
    if vis:
        images_dis = images_dis[0]
        _, c1, c2, c3 = model(images_dis)
        I_test = utils.make_grid(images_dis, nrow=6, normalize=True, scale_each=True)
        writer.add_image('test/image', I_test)

        if c1 is not None:
            attn1 = visualize_attn_softmax(I_test, c1, up_factor=4, nrow=6)
            writer.add_image('test/attention_map_1', attn1, epoch)
        if c2 is not None:
            attn2 = visualize_attn_softmax(I_test, c2, up_factor=8, nrow=6)
            writer.add_image('test/attention_map_2', attn2, epoch)
        if c3 is not None:
            attn3 = visualize_attn_softmax(I_test, c3, up_factor=16, nrow=6)
            writer.add_image('test/attention_map_3', attn3, epoch)
            writer.add_image('test/2', attn2, epoch)


def main(args):
    # load model
    model = AttnVGG_before(10)
    model = nn.DataParallel(model, device_ids=[0])
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict['state_dict'])

    # create loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    testdir = os.path.join(args.dataset_path)
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)
    writer = SummaryWriter(args.logs)
    test(model, test_loader, writer, vis=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provide arguments for inferences')
    parser.add_argument('--dataset_path', default='/scratch/Datasets/Tobacco')
    parser.add_argument('--model_path', default='model_best.pth.tar')
    parser.add_argument('--logs', default='Retrained_VGG_Tobaco_TrainAttnLayer')
    args = parser.parse_args()
    main(args)

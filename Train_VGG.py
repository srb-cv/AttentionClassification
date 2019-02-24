
import torch.nn as nn
from datasets.models import vgg


def vgg_512fc(num_classes):

    model = vgg.vgg16_bn(pretrained=True)

    # for param in model.parameters():
    #     param.requires_grad = False


    model.classifier._modules['6'] = nn.Linear(4096, 512, bias=True)

    model.classifier._modules['7'] = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    #print(model)

    return model
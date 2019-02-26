import torch.nn as nn
from datasets.models import vgg
import torch


def vgg_512fc(num_classes):

    model = vgg.vgg16_bn(pretrained=True)

    # for param in model.parameters():
    #     param.requires_grad = False


    model.classifier._modules['6'] = nn.Linear(4096, 512, bias=True)

    model.classifier._modules['7'] = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    #print(model)

    return model


def vgg_attn_from_cdip(num_classes, resume_path):
    import model1
    model = model1.AttnVGG_before(num_classes=10, attention=True, normalize_attn=False)
    checkpoint = torch.load(resume_path)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
        'state_dict'].items()}
    model.load_state_dict(state_dict)
    model.classify = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
    return model


def vgg_non_attn_from_tobacco(num_classes, resume_path):
    model = model1.AttnVGG_before(num_classes=10, attention=False, normalize_attn=False)
    checkpoint = torch.load(resume_path)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
        'state_dict'].items()}
    model.load_state_dict(state_dict)
    model.classify = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
    return model


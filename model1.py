import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, LinearAttentionBlock, ProjectorBlock
from initialize import *
from torchvision import models
import torch.nn as nn

'''
attention before max-pooling
'''


class AttnVGG_before(nn.Module):
    def __init__(self, num_classes, attention=True, normalize_attn=True, init='default'):
        super(AttnVGG_before, self).__init__()
        self.attention = attention
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        #         self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        #         self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size/32), padding=0, bias=True)
#         self.dense1 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding=0, bias=True)
#         self.dense2 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0, bias=True)
        self.dense1= nn.Linear(in_features=512 * 7*7, out_features=4096, bias=True)
        self.dense2= nn.Linear(in_features=4096, out_features=4096, bias=True)
        for param in self.parameters():
            param.requires_grad = False
        self.dense3 = nn.Linear(in_features=4096, out_features=512, bias=True)

        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            print("Initializing Default weights")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # feed forward
        l4 = self.conv_block1(x)
        x = F.max_pool2d(l4, kernel_size=2, stride=2, padding=0)  # /2
        l5 = self.conv_block2(x)
        x = F.max_pool2d(l5, kernel_size=2, stride=2, padding=0)  # /2
        l1 = self.conv_block3(x)  # /1
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)  # /2
        l2 = self.conv_block4(x)  # /2
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)  # /4
        l3 = self.conv_block5(x)  # /4
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0)  # /8
        #         x = self.conv_block6(x) # /32
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        f1 = self.dense1(x)  # batch_sizex512x1x1
        f2 = self.dense2(f1)
        g = self.dense3(f2)
        #g=f2
        # pay attention
        if self.attention:
            g=g.unsqueeze(2).unsqueeze(3)
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1, g2, g3), dim=1)  # batch_sizexC
            # classification layer
            x = self.classify(g)  # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return [x, c1, c2, c3]


    def copy_weights_vgg16(self):

        model = models.vgg16_bn(pretrained=True)

        for l1, l2 in zip(model.features[:6], self.conv_block1.op):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                # print(l1,l2)
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data

        for l1, l2 in zip(model.features[7:13], self.conv_block2.op):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data

        for l1, l2 in zip(model.features[14:23], self.conv_block3.op):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data

        for l1, l2 in zip(model.features[24:33], self.conv_block4.op):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data

        for l1, l2 in zip(model.features[34:43], self.conv_block5.op):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data

        l1 = model.classifier[0]
        l2 = self.dense1
        if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
            #l2.weight.data = l1.weight.reshape(l2.weight.shape).data
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data

        l1 = model.classifier[3]
        l2 = self.dense2
        if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
            #l2.weight.data = l1.weight.reshape(l2.weight.shape).data
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data


        # l1 = model.classifier[6]
        # l2 = self.classify
        # if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
        #     #l2.weight.data = l1.weight.reshape(l2.weight.shape).data
        #     l2.weight.data = l1.weight.data
        #     l2.bias.data = l1.bias.data



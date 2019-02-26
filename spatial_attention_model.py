import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, SpatialAttentionBlock, ProjectorBlock
from initialize import *
from torchvision import models
import torch.nn as nn
from Train_VGG import vgg_512fc
from collections import OrderedDict
'''
attention after max-pooling
'''



class AttnVGG_spatial(nn.Module):
    def __init__(self, num_classes, attention=True, normalize_attn=True, init='default', reduced_attention = False):
        super(AttnVGG_spatial, self).__init__()
        self.attention = attention
        self.reduced_attention = reduced_attention
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
        self.dense3 = nn.Linear(in_features=4096, out_features=512, bias=True)

        for param in self.parameters():
            param.requires_grad = False


        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = SpatialAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = SpatialAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = SpatialAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention and not reduced_attention:
            self.classify = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
        elif self.attention and reduced_attention:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
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
        x = self.conv_block1(x)
        l1 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # /2
        x = self.conv_block2(l1)
        l2 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # /2
        x = self.conv_block3(l2)  # /1
        l3 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # /2
        x = self.conv_block4(l3)  # /2
        l4 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # /4
        x = self.conv_block5(l4)  # /4
        l5 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # /8
        #         x = self.conv_block6(x) # /32
        x = self.avgpool(l5)
        x = x.view(x.size(0), -1)

        f1 = self.dense1(x)  # batch_sizex512x1x1
        f2 = self.dense2(f1)
        g = self.dense3(f2)



        # pay attention
        if self.attention and not self.reduced_attention:
            g=g.unsqueeze(2).unsqueeze(3)
            c1, g1 = self.attn1(self.projector(l3), g)
            c2, g2 = self.attn2(l4, g)
            c3, g3 = self.attn3(l5, g)
            g = torch.cat((g1, g2, g3), dim=1)  # batch_sizexC
            # classification layer
            x = self.classify(g)  # batch_sizexnum_classes

        elif self.attention and self.reduced_attention:
            g = g.unsqueeze(2).unsqueeze(3)
            c1, g1 = None, None
            c2, g2 = None, None
            c3, g3 = self.attn3(l5, g)
            #g = torch.cat((g2, g3), dim=1)  # batch_sizexC
            # classification layer
            x = self.classify(g3)  # batch_sizexnum_classes

        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return [x, c1, c2, c3]

    def copy_weights_vgg16(self, model_path_to_copy_weigths):
        #model = models.vgg16_bn(pretrained=True)
        model = vgg_512fc(num_classes=16)
        model = self.load_weight(model, model_path_to_copy_weigths)

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


        l1 = model.classifier[6]
        l2 = self.dense3
        if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
            #l2.weight.data = l1.weight.reshape(l2.weight.shape).data
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data

    def copy_weights_no_attn_vgg16(self, model_path_to_copy_weigths):
        # model = models.vgg16_bn(pretrained=True)
        model = AttnVGG_spatial(num_classes=10, attention=False, normalize_attn=False, reduced_attention=True)
        model = self.load_weight(model, model_path_to_copy_weigths)

        for l1, l2 in zip(model.conv_block1.op, self.conv_block1.op):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                # print(l1,l2)
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data

        for l1, l2 in zip(model.conv_block2.op, self.conv_block2.op):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data

        for l1, l2 in zip(model.conv_block3.op, self.conv_block3.op):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data

        for l1, l2 in zip(model.conv_block4.op, self.conv_block4.op):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data

        for l1, l2 in zip(model.conv_block5.op, self.conv_block5.op):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data

        l1 = model.dense1
        l2 = self.dense1
        if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
            # l2.weight.data = l1.weight.reshape(l2.weight.shape).data
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data

        l1 = model.dense2
        l2 = self.dense2
        if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
            # l2.weight.data = l1.weight.reshape(l2.weight.shape).data
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data

        l1 = model.dense3
        l2 = self.dense3
        if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
            # l2.weight.data = l1.weight.reshape(l2.weight.shape).data
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data

    def load_weight(self, model, path):
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

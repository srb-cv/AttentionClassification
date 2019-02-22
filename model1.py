import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, LinearAttentionBlock, ProjectorBlock
from initialize import *

'''
attention before max-pooling
'''


class AttnVGG_before(nn.Module):
    def __init__(self, num_classes, attention=True, normalize_attn=True, init='default'):
        super(AttnVGG_before, self).__init__()
        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        #         self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        #         self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size/32), padding=0, bias=True)
        self.dense1 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding=0, bias=True)
        self.dense2 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0, bias=True)
        self.dense3 = nn.Conv2d(in_channels=4096, out_channels=512, kernel_size=1, padding=0, bias=True)

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
        f1 = self.dense1(x)  # batch_sizex512x1x1
        f2 = self.dense2(f1)
        g = self.dense3(f2)
        # pay attention
        if self.attention:
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
        #return x
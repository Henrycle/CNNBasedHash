import torch.nn as nn
from model.conv_bn_relu import ConvBNRelu
from options import HiDDenConfiguration

class Deephash(nn.Module):

    def __init__(self,config: HiDDenConfiguration):
        super(Deephash, self).__init__()
        self.block_num = config.blocks_num
        self.first_channel = config.first_block

        layers = [ConvBNRelu(3,self.first_channel)]
        layers.append(nn.MaxPool2d(2))

        for _ in range(self.block_num - 2):
            layer = ConvBNRelu(self.first_channel,2*self.first_channel)
            layer_pool = nn.MaxPool2d(2)
            self.first_channel = 2*self.first_channel
            layers.append(layer)
            layers.append(layer_pool)

        layers.append(ConvBNRelu(self.first_channel,2*self.first_channel))

        self.hidden_layers = nn.Sequential(*layers)

        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(2*self.first_channel, config.L)

    def forward(self, image):

        out_before_avg = self.hidden_layers(image)
        out_after_avg = self.avgpool(out_before_avg)
        out = out_after_avg.view(out_after_avg.size(0), -1)
        hash = self.fc(out)

        return out_before_avg, out_after_avg, hash
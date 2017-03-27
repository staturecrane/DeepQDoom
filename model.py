import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class DoomConv(nn.Module):
    def __init__(self, channels, output_size):
        super(DoomConv, self).__init__()
        self.channels = channels
        self.output_size = output_size
        self.hidden_size = 128

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 1)
        )

        self.lstm = nn.LSTMCell(128, 128)

        self.output_lin = nn.Linear(128, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, hidden, state):
        conv_output = self.conv(x).view(1, 128)
        hidden, state = self.lstm(conv_output, (hidden, state))
        output = self.output_lin(hidden)
        return output, hidden, state

    def initHidden(self):
        return (
            Variable(torch.FloatTensor(1, self.hidden_size).cuda().zero_()),
            Variable(torch.FloatTensor(1, self.hidden_size).cuda().zero_())
        )

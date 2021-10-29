import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(1, 0), dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self, temperature):
        super(AttentionGate, self).__init__()
        kernel_size = (5, 1)
        self.temperature = temperature
        self.compress = ZPool()
        # self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(2, 0), relu=False)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        # print(x.shape, 'ty1')
        x_compress = self.compress(x)
        # print(x_compress.shape, 'Z_pooling')
        x_out = self.conv(x_compress)
        # print(x_out.shape, 'Conv+BN+RelU')
        # scale = torch.softmax(x_out/self.temperature, 1)
        scale = torch.sigmoid(x_out)
        # print((x*scale).shape, 'ty4')
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False, temperature=34):
        super(TripletAttention, self).__init__()

        self.cw = AttentionGate(temperature)
        self.hc = AttentionGate(temperature)
        self.no_spatial = no_spatial

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # initialization
        # self.w1 = torch.nn.init.normal_(self.w1)
        # self.w2 = torch.nn.init.normal_(self.w2)
        # self.w3 = torch.nn.init.normal_(self.w3)
        self.w1.data.fill_(1/3)
        self.w2.data.fill_(1/3)
        self.w3.data.fill_(1/3)

        if not no_spatial:
            self.hw = AttentionGate(temperature)

    def update_temperature(self):
        self.cw.updata_temperature()
        self.hc.updata_temperature()
        self.hw.updata_temperature()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        # print(x_out1.shape, 'ty44')
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            # print(x_out.shape, 'ty55')
            # x_out = x_out11
            # x_out = 1/3 * (x_out + x_out11 + x_out21)
            # x_out = 4 * x_out + 5 * x_out11 + 6 * x_out21
            x_out = self.w1 * x_out + self.w2 * x_out11 + self.w3 * x_out21
            # print(self.w1, self.w2, self.w3, 'w1,w2,w3')
            # print(x_out.shape, 'ty22')
        else:
            x_out = self.w1 * x_out11 + self.w2 * x_out21
        # return x_out, self.w1, self.w2, self.w3
        return x_out


# if __name__ == '__main__':
#     x = torch.randn(60, 64,  171, 40)
#     model = TripletAttention()
#     x = x.to('cuda:0')
#     model.to('cuda')
#     out = model(x)
#     print(model)
#     print(out[0].shape)
#     print(out[1])
#     print(out[2])
#     print(out[3])



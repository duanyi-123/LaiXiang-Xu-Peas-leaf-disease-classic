import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import DeformConv2d


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LightweightSelfAttention(nn.Module):
    """轻量化自注意力模块，适用于MobileNetV3"""

    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            h_sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            h_sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)

        # 空间注意力
        sa = self.spatial_attention(x)

        # 结合两种注意力
        return x * ca * sa


class AdditiveAttention(nn.Module):
    """加性注意力机制，用于第7层和第9层"""

    def __init__(self, in_channels, key_channels=None):
        super().__init__()
        if key_channels is None:
            key_channels = in_channels // 8
        self.key_channels = key_channels

        # 查询、键、值的卷积
        self.query_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 缩放因子
        self.scale = 1.0 / math.sqrt(key_channels)

        # 输出投影
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 计算查询、键、值
        query = self.query_conv(x).view(batch_size, self.key_channels, -1)  # B x C' x (H*W)
        key = self.key_conv(x).view(batch_size, self.key_channels, -1)  # B x C' x (H*W)
        value = self.value_conv(x).view(batch_size, channels, -1)  # B x C x (H*W)

        # 计算注意力得分
        attention = torch.bmm(query.permute(0, 2, 1), key)  # B x (H*W) x (H*W)
        attention = F.softmax(attention * self.scale, dim=-1)

        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, channels, height, width)

        # 残差连接
        return self.gamma * out + x


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformableConv2d, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)
        self.deform_conv = DeformConv2d(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding)

    def forward(self, x):
        offsets = self.offset_conv(x)
        return self.deform_conv(x, offsets)


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, layer_idx=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        self.layer_idx = layer_idx

        # 判断是否为第3、4、8层（可变形卷积）
        self.use_deform = layer_idx in [3, 4, 8]
        # 判断是否为第6、10层（自注意力）
        self.use_self_attn = layer_idx in [6, 10]
        # 判断是否为第7、9层（加性注意力）
        self.use_additive_attn = layer_idx in [7, 9]

        if inp == hidden_dim:
            if self.use_deform:
                self.conv = nn.Sequential(
                    # dw
                    DeformableConv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2),
                    nn.BatchNorm2d(hidden_dim),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # Squeeze-and-Excite
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                              bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # Squeeze-and-Excite
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
        else:
            if self.use_deform:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # dw
                    DeformableConv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2),
                    nn.BatchNorm2d(hidden_dim),
                    # Squeeze-and-Excite
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                              bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    # Squeeze-and-Excite
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )

        # 在第6、10层后添加自注意力
        if self.use_self_attn:
            self.self_attn = LightweightSelfAttention(oup)

        # 在第7、9层后添加加性注意力
        if self.use_additive_attn:
            self.additive_attn = AdditiveAttention(oup)

    def forward(self, x):
        out = self.conv(x)
        if self.use_self_attn:
            out = self.self_attn(out)
        if self.use_additive_attn:
            out = self.additive_attn(out)
        if self.identity:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=5, width_mult=1.):
        super(MobileNetV3, self).__init__()
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        block = InvertedResidual
        for layer_idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs, 1):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, layer_idx))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[
            mode]

        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, DeformConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


def mobilenetv3_small(**kwargs):
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 1, 0, 2],  # layer 1
        [3, 4.5, 24, 0, 0, 2],  # layer 2
        [3, 3.67, 24, 0, 0, 1],  # layer 3 (可变形卷积)
        [5, 4, 40, 1, 1, 2],  # layer 4 (可变形卷积)
        [5, 6, 40, 1, 1, 1],  # layer 5
        [5, 6, 40, 1, 1, 1],  # layer 6 (自注意力)
        [5, 3, 48, 1, 1, 1],  # layer 7 (加性注意力)
        [5, 3, 48, 1, 1, 1],  # layer 8 (可变形卷积)
        [5, 6, 96, 1, 1, 2],  # layer 9 (加性注意力)
        [5, 6, 96, 1, 1, 1],  # layer 10 (自注意力)
        [5, 6, 96, 1, 1, 1],  # layer 11
    ]
    return MobileNetV3(cfgs, mode='small', **kwargs)
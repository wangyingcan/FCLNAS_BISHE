# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.


from utils import *
from collections import OrderedDict


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        ZeroLayer.__name__: ZeroLayer,
        ResNetBlock.__name__: ResNetBlock,
        DenseNetBlock.__name__: DenseNetBlock,
        SEBlock.__name__: SEBlock,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)

'''
    2D通用layer
'''
class My2DLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False

'''
    标准卷积层
'''
class ConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': ConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(ConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)

    def get_flops(self, x):
        return count_conv_flop(self.conv, x), self.forward(x)

'''
    Depthwise + Pointwise 的可分离卷积
'''
class DepthConvLayer(My2DLayer):
    # Depth-wise Separable Convolutions

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(DepthConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['depth_conv'] = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.in_channels, bias=False
        )
        weight_dict['point_conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            return '%dx%d_DilatedDepthConv' % (kernel_size[0], kernel_size[1])
        else:
            return '%dx%d_DepthConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': DepthConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(DepthConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)

    def get_flops(self, x):
        depth_flop = count_conv_flop(self.depth_conv, x)
        x = self.depth_conv(x)
        point_flop = count_conv_flop(self.point_conv, x)
        x = self.point_conv(x)
        return depth_flop + point_flop, self.forward(x)

'''
    池化层
'''
class PoolingLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 pool_type, kernel_size=2, stride=2,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        super(PoolingLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self.pool_type == 'avg':
            weight_dict['pool'] = nn.AvgPool2d(
                self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False
            )
        elif self.pool_type == 'max':
            weight_dict['pool'] = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_%sPool' % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        return {
            'name': PoolingLayer.__name__,
            'pool_type': self.pool_type,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            **super(PoolingLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

'''
    恒等映射（可能带 BN / 激活）
'''
class IdentityLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return 'Identity'

    @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

'''
    全连接层
'''
class LinearLayer(MyModule):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm1d(in_features)
            else:
                modules['bn'] = nn.BatchNorm1d(out_features)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # linear
        modules['weight'] = {'linear': nn.Linear(self.in_features, self.out_features, self.bias)}

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_Linear' % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

    def get_flops(self, x):
        return self.linear.weight.numel(), self.forward(x)

    # returns the number of elements in an array

    @staticmethod
    def is_zero_layer():
        return False

'''
    MobileNet 的 Inverted Residual 卷积块
'''
class MBInvertedConvLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', nn.ReLU6(inplace=True)),
        ]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_MBConv%d' % (self.kernel_size, self.kernel_size, self.expand_ratio)

    @property
    def config(self):
        return {
            'name': MBInvertedConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    def get_flops(self, x):
        if self.inverted_bottleneck:
            flop1 = count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        else:
            flop1 = 0

        flop2 = count_conv_flop(self.depth_conv.conv, x)
        x = self.depth_conv(x)

        flop3 = count_conv_flop(self.point_linear.conv, x)
        x = self.point_linear(x)

        return flop1 + flop2 + flop3, x

    @staticmethod
    def is_zero_layer():
        return False
'''
    输出全 0 的“空操作层”
'''
class ZeroLayer(MyModule):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding

    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return True

################## 新增 ##################
# ResNetBlock
class ResNetBlock(MyModule):
    """标准 ResNet BasicBlock，支持 stride>1，通过 1x1 shortcut 对齐形状。"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = 3
        self.expand_ratio = 1
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=stride, act_func='relu6')
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, act_func=None)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvLayer(
                in_channels, out_channels, kernel_size=1, stride=stride, use_bn=True, act_func=None
            )
        else:
            self.shortcut = IdentityLayer(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.act(out)

    def get_flops(self, x):
        flop1, out1 = self.conv1.get_flops(x)
        flop2, out2 = self.conv2.get_flops(out1)
        flop_sc, _ = self.shortcut.get_flops(x)
        out = out2 + self.shortcut(x)
        return flop1 + flop2 + flop_sc, out

    @staticmethod
    def is_zero_layer():
        return False

    @property
    def module_str(self):
        return f"ResNetBlock(k3_s{self.stride}_c{self.out_channels})"

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "stride": self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ResNetBlock(
            in_channels=config.get("in_channels"),
            out_channels=config.get("out_channels"),
            stride=config.get("stride", 1),
        )

class DenseNetBlock(MyModule):
    """轻量 DenseNet block：堆叠若干层后用 1x1 conv 压缩，并可选下采样。"""
    def __init__(self, in_channels, out_channels, stride=1, growth_rate=12, num_layers=6):
        super(DenseNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = 3
        self.expand_ratio = 1
        self.layers = nn.ModuleList()
        cur_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(
                ConvLayer(cur_channels, growth_rate, kernel_size=3, stride=1, act_func='relu6')
            )
            cur_channels += growth_rate
        self.transition = ConvLayer(cur_channels, out_channels, kernel_size=1, stride=1, act_func=None)
        self.downsample = nn.Identity() if stride == 1 else nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        features = x
        for layer in self.layers:
            new_feat = layer(features)
            features = torch.cat([features, new_feat], dim=1)
        out = self.transition(features)
        out = self.downsample(out)
        return out

    def get_flops(self, x):
        total_flop = 0
        features = x
        for layer in self.layers:
            flop, new_feat = layer.get_flops(features)
            total_flop += flop
            features = torch.cat([features, new_feat], dim=1)
        flop_trans, out = self.transition.get_flops(features)
        total_flop += flop_trans
        out = self.downsample(out)
        # downsample (avgpool) FLOPs 近似为 0，这里忽略
        return total_flop, out

    @staticmethod
    def is_zero_layer():
        return False

    @property
    def module_str(self):
        return f"DenseNetBlock(k3_layers{len(self.layers)}_growth{self.layers[0].out_channels}_s{self.stride}_out{self.out_channels})"

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "stride": self.stride,
            "growth_rate": self.layers[0].out_channels if len(self.layers) > 0 else None,
            "num_layers": len(self.layers),
        }

    @staticmethod
    def build_from_config(config):
        return DenseNetBlock(
            in_channels=config.get("in_channels"),
            out_channels=config.get("out_channels"),
            stride=config.get("stride", 1),
            growth_rate=config.get("growth_rate", 12),
            num_layers=config.get("num_layers", 6),
        )

class SqueezeExcitation(MyModule):
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, squeeze_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.avg_pool(x))
        return x * scale

class SEBlock(MyModule):
    """3x3 两层卷积 + SE + 残差"""
    def __init__(self, in_channels, out_channels, stride=1, reduction=4):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = 3
        self.expand_ratio = 1
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=stride, act_func='relu6')
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, act_func=None)
        self.se = SqueezeExcitation(out_channels, reduction)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvLayer(
                in_channels, out_channels, kernel_size=1, stride=stride, use_bn=True, act_func=None
            )
        else:
            self.shortcut = IdentityLayer(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        return self.act(out)

    def get_flops(self, x):
        flop1, out1 = self.conv1.get_flops(x)
        flop2, out2 = self.conv2.get_flops(out1)
        # SE 部分 FLOPs 估计（两个 1x1 conv）
        n, c, h, w = out2.size()
        squeeze_c = max(1, c // self.se.fc[0].out_channels) if hasattr(self.se.fc[0], "out_channels") else max(1, c // 4)
        se_flop = c * h * w  # global avg
        se_flop += c * squeeze_c + squeeze_c * c  # 两个 1x1 conv 近似
        out = self.se(out2)
        flop_sc, _ = self.shortcut.get_flops(x)
        out = out + self.shortcut(x)
        return flop1 + flop2 + se_flop + flop_sc, self.act(out)

    @staticmethod
    def is_zero_layer():
        return False

    @property
    def module_str(self):
        return f"SEBlock(k3_s{self.stride}_c{self.out_channels})"

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "stride": self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return SEBlock(
            in_channels=config.get("in_channels"),
            out_channels=config.get("out_channels"),
            stride=config.get("stride", 1),
        )

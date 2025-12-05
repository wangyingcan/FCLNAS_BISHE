# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from queue import Queue
import copy

from modules.mix_op import *
from models.normal_nets.proxyless_nets import *
from utils import LatencyEstimator
from utils.pytorch_utils import compute_gpu_latency_ms_pytorch, compute_cpu_latency_ms_pytorch, lat_table


class SuperProxylessNASNets(ProxylessNASNets):

    # 超网网络结构的组成
    def __init__(self, width_stages, n_cell_stages, conv_candidates, stride_stages,
                 n_classes=10, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0, inference_device='gpu'):
        print('SuperProxylessNASNets初始化开始...')
        self._redundant_modules = None
        self._unused_modules = None
        self.latency_model = LatencyEstimator(device=inference_device)

        input_channel = make_divisible(32 * width_mult, 8)      # 输入通道
        first_cell_width = make_divisible(16 * width_mult, 8)      # 输出通道
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)

        # 输入层  first conv layer  固定的第一层卷积层（将3通道图片映射到input_channel）
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )

        # 第一块  first block  仅含一个选择的MixedEdge
        first_block_conv = MixedEdge(candidate_ops=build_candidate_ops(
            ['3x3_MBConv1'],
            input_channel, first_cell_width, 1, 'weight_bn_act',
        ), )
        if first_block_conv.n_choices == 1:
            first_block_conv = first_block_conv.candidate_ops[0]        # 仅有一个ops，所以相当于固定第一层
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_cell_width

        # blocks 后续block构建
        blocks = [first_block]
        '''
            width_stages: 24,40,80,96,192,320
            n_cell_stages: 2,3,4,3,4,3
            stride_stages: 1,1,2,1,2,1
        '''
        for stage_cnt, (width, n_cell, s) in enumerate(zip(width_stages, n_cell_stages, stride_stages)):

            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                
                if stride == 1 and input_channel == width:
                    modified_conv_candidates = conv_candidates# + ['Zero']
                else:
                    modified_conv_candidates = conv_candidates

                # 核心的搜索组件 MixedEdge
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, input_channel, width, stride, 'weight_bn_act',
                ), )  # output_channel = width

                if stride == 1 and input_channel == width:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None

                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_channel = width

        # 特征融合层 + 分类头 feature mix layer
        last_channel = make_divisible(400 * width_mult, 8) if width_mult > 1.0 else 400
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
        )

        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
        super(SuperProxylessNASNets, self).__init__(first_conv, blocks, feature_mix_layer, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    """ 权重参数 weight parameters, arch_parameters & binary gates """
    def architecture_parameters(self):
        for name, param in self.named_parameters():     # 获取模型的所有可训练参数及其对应的名称
            if 'AP_path_alpha' in name:
                yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param

    """ 架构参数 architecture parameters related methods """
    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support `set_arch_param_grad()`')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    """ training related methods """
    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')

    def set_active_via_net(self, net):
        assert isinstance(net, SuperProxylessNASNets)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    def expected_flops(self, x):
        expected_flops = 0
        # first conv
        flop, x = self.first_conv.get_flops(x)
        expected_flops += flop
        # blocks
        for block in self.blocks:
            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                delta_flop, x = block.get_flops(x)
                expected_flops = expected_flops + delta_flop
                continue

            if block.shortcut is None:
                shortcut_flop = 0
            else:
                shortcut_flop, _ = block.shortcut.get_flops(x)
            expected_flops = expected_flops + shortcut_flop

            probs_over_ops = mb_conv.current_prob_over_ops
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                op_flops, _ = op.get_flops(x)
                expected_flops = expected_flops + op_flops * probs_over_ops[i]
            x = block(x)
        # feature mix layer
        delta_flop, x = self.feature_mix_layer.get_flops(x)
        expected_flops = expected_flops + delta_flop
        # classifier
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        delta_flop, x = self.classifier.get_flops(x)
        expected_flops = expected_flops + delta_flop
        return expected_flops

    def build_latency_table(self, inference_device='gpu'):
        # model.eval()
        # model.cpu()
        # model.cuda()
        if inference_device == 'gpu':
            compute_latency = compute_gpu_latency_ms_pytorch
        else:
            compute_latency = compute_cpu_latency_ms_pytorch
        latency_table = lat_table(hardware=inference_device)

        latency, output = compute_latency(self.first_conv, (128, 3, 32, 32))
        latency_table.record(ltype='Conv', _input=[3, 32, 32], output=[self.first_conv.out_channels, 32, 32],
                             latency=latency)
        print('first conv')
        fsize = 32

        # blocks
        for block_num, block in enumerate(self.blocks):
            shortcut = block.shortcut
            if shortcut is None or shortcut.is_zero_layer():
                idskip = 0
            else:
                idskip = 1
            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                if not mb_conv.is_zero_layer():
                    out_fz = fsize // mb_conv.stride
                    op_latency, output = compute_latency(mb_conv, (128, mb_conv.in_channels, fsize, fsize))
                    latency_table.record(ltype='expanded_conv', _input=[mb_conv.in_channels, fsize, fsize],
                                         output=[mb_conv.out_channels, out_fz, out_fz], expand=mb_conv.expand_ratio,
                                         kernel=mb_conv.kernel_size, stride=mb_conv.stride, idskip=idskip,
                                         latency=op_latency)
                    fsize = out_fz
                continue

            out_fsize = fsize
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                out_fsize = fsize // op.stride

                op_latency, output = compute_latency(op, (128, op.in_channels, fsize, fsize))
                latency_table.record(ltype='expanded_conv', _input=[op.in_channels, fsize, fsize],
                                     output=[op.out_channels, out_fsize, out_fsize], expand=op.expand_ratio,
                                     kernel=op.kernel_size, stride=op.stride, idskip=idskip,
                                     latency=op_latency)
            fsize = out_fsize
            print(block_num, ' block')

        latency, output = compute_latency(self.feature_mix_layer, (128, self.feature_mix_layer.in_channels, 8, 8))
        latency_table.record(ltype='Conv_1', _input=[self.feature_mix_layer.in_channels, 8, 8],
                             output=[self.feature_mix_layer.out_channels, 8, 8], latency=latency)
        print('feature mix layer')

        latency, output = compute_latency(self.classifier, (128, self.classifier.in_features))
        latency_table.record(ltype='Logits', _input=[self.classifier.in_features],
                             output=[self.classifier.out_features], latency=latency)
        print('classifier')

    def expected_latency(self, ):
        expected_latency = 0
        # first conv
        expected_latency += self.latency_model.predict('Conv', [3, 32, 32], [self.first_conv.out_channels, 32, 32])
        # feature mix layer
        expected_latency += self.latency_model.predict(
            'Conv_1', [self.feature_mix_layer.in_channels, 8, 8], [self.feature_mix_layer.out_channels, 8, 8]
        )
        # classifier
        expected_latency += self.latency_model.predict(
            'Logits', [self.classifier.in_features], [self.classifier.out_features]  # 1000
        )
        # blocks
        fsize = 32
        for block in self.blocks:
            shortcut = block.shortcut
            if shortcut is None or shortcut.is_zero_layer():
                idskip = 0
            else:
                idskip = 1

            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                if not mb_conv.is_zero_layer():
                    out_fz = fsize // mb_conv.stride
                    op_latency = self.latency_model.predict(
                        'expanded_conv', [mb_conv.in_channels, fsize, fsize], [mb_conv.out_channels, out_fz, out_fz],
                        expand=mb_conv.expand_ratio, kernel=mb_conv.kernel_size, stride=mb_conv.stride, idskip=idskip
                    )
                    expected_latency = expected_latency + op_latency
                    fsize = out_fz
                continue

            probs_over_ops = mb_conv.current_prob_over_ops
            out_fsize = fsize
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                out_fsize = fsize // op.stride
                op_latency = self.latency_model.predict(
                    'expanded_conv', [op.in_channels, fsize, fsize],
                    [op.out_channels, out_fsize, out_fsize],
                    expand=op.expand_ratio, kernel=op.kernel_size, stride=op.stride, idskip=idskip
                )
                expected_latency = expected_latency + op_latency * probs_over_ops[i]
            fsize = out_fsize
        return expected_latency

    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        return ProxylessNASNets(self.first_conv, list(self.blocks), self.feature_mix_layer, self.classifier)

    print('SuperProxylessNASNets初始化完成...')

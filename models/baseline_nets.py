import copy

import torch.nn as nn
from torchvision import models

from utils.my_modules import MyNetwork


class BaselineResNet(MyNetwork):
    """
    轻量封装 torchvision ResNet，使其符合 RunManager 所需接口。
    """

    def __init__(self, arch: str = "resnet18", num_classes: int = 100, pretrained: bool = False):
        super().__init__()
        if not hasattr(models, arch):
            raise ValueError(f"torchvision.models 中不存在 {arch}")
        builder = getattr(models, arch)
        try:
            self.backbone = builder(weights="IMAGENET1K_V1" if pretrained else None)
        except TypeError:
            # 兼容旧版 torchvision
            self.backbone = builder(pretrained=pretrained)

        # 替换分类头
        if not hasattr(self.backbone, "fc"):
            raise ValueError(f"{arch} 缺少 fc 层，当前 baseline 仅支持标准 ResNet")
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        self.arch = arch
        self.num_classes = num_classes
        self.pretrained = pretrained

    def forward(self, x):
        return self.backbone(x)

    @property
    def module_str(self):
        return f"BaselineResNet({self.arch})"

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "arch": self.arch,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
        }

    @staticmethod
    def build_from_config(config):
        return BaselineResNet(
            arch=config.get("arch", "resnet18"),
            num_classes=config.get("num_classes", 100),
            pretrained=config.get("pretrained", False),
        )

    def get_flops(self, x):
        # baseline 只用于对比训练，FLOPs 估计可选，返回 0 以兼容上层接口
        return 0, None

    def set_bn_param(self, momentum, eps):
        # 复用 MyNetwork 中的 BN 设置逻辑
        return super().set_bn_param(momentum, eps)

    def init_model(self, model_init, init_div_groups=False):
        # 复用默认初始化逻辑覆盖 resnet 权重
        return super().init_model(model_init, init_div_groups)

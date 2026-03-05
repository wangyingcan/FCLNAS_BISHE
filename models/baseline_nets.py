import torch.nn as nn
from torchvision import models

from utils.my_modules import MyNetwork


class BaselineResNet(MyNetwork):
    """
    轻量封装 torchvision 分类骨干网络（当前重点支持 resnet* / mobilenet_v2），
    使其符合 RunManager 所需接口。
    """

    def __init__(self, arch: str = "resnet18", num_classes: int = 100, pretrained: bool = False):
        super().__init__()
        if not hasattr(models, arch):
            raise ValueError(f"torchvision.models 中不存在 {arch}")
        builder = getattr(models, arch)
        self.backbone = self._build_backbone(builder, arch, pretrained)
        self._replace_classifier_head(self.backbone, arch, num_classes)

        self.arch = arch
        self.num_classes = num_classes
        self.pretrained = pretrained

    @staticmethod
    def _build_backbone(builder, arch: str, pretrained: bool):
        if pretrained:
            # 新版 torchvision 的 enum 权重优先；映射不到时回退到旧接口。
            enum_name_map = {
                "resnet18": "ResNet18_Weights",
                "resnet34": "ResNet34_Weights",
                "resnet50": "ResNet50_Weights",
                "mobilenet_v2": "MobileNet_V2_Weights",
            }
            enum_name = enum_name_map.get(arch)
            if enum_name is not None and hasattr(models, enum_name):
                try:
                    enum_cls = getattr(models, enum_name)
                    return builder(weights=enum_cls.DEFAULT)
                except Exception:
                    pass
            for kwargs in ({"weights": "IMAGENET1K_V1"}, {"pretrained": True}, {}):
                try:
                    return builder(**kwargs)
                except Exception:
                    continue
            raise RuntimeError(f"无法构建 pretrained backbone: {arch}")

        for kwargs in ({"weights": None}, {"pretrained": False}, {}):
            try:
                return builder(**kwargs)
            except Exception:
                continue
        raise RuntimeError(f"无法构建 backbone: {arch}")

    @staticmethod
    def _replace_classifier_head(backbone, arch: str, num_classes: int):
        # ResNet 系列：backbone.fc
        if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_classes)
            return

        # MobileNet / EfficientNet 系列：backbone.classifier
        if hasattr(backbone, "classifier"):
            classifier = backbone.classifier
            if isinstance(classifier, nn.Linear):
                in_features = classifier.in_features
                backbone.classifier = nn.Linear(in_features, num_classes)
                return
            if isinstance(classifier, nn.Sequential):
                for idx in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[idx], nn.Linear):
                        in_features = classifier[idx].in_features
                        classifier[idx] = nn.Linear(in_features, num_classes)
                        return

        raise ValueError(
            f"{arch} 缺少可替换的分类头（fc/classifier），"
            "当前 baseline 仅支持标准分类骨干（如 resnet* / mobilenet_v2）"
        )

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

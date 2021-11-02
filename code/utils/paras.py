# 计算参数量的
import torch
import torchvision
# from model.model_v2 import MobileNetV2
# from model.MobResNet import MobResNet18
from pytorch_model_summary import summary
from models.Mob_ResNet import MobResNet18
# model = efficientnet_b0(pretrained=False)
model = MobResNet18(8)
print(model)

dummy_input = torch.randn(1, 3, 224, 224)
print(summary(model, dummy_input, show_input=False, show_hierarchical=False))
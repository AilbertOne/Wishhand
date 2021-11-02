# for i in range(4):
#     stride = 1 if i ==0 else 1
#     print("i = ",i, "stride = ",stride)
# 打印网络结构
# import torchsummary
# from model.model import *
# net = ResMobNet34().cuda()
# print(net)

# 测试图片
from model.modeladd import ResMobNet50
import torch
net = ResMobNet50(3)
fake_img = torch.randn((1,3,224,224), dtype=torch.float32)
#
# #
output = net(fake_img)
print(output)
# 测试特征矩阵相加
# x = torch.randn((1, 3, 3, 3), dtype=torch.float32)
# y = torch.randn((1, 3, 3, 3), dtype=torch.float32)
# print(x.size())
# print(x)
# print(y.size())
# print(y)
# z = torch.add(x, y)
# print(z.size())
# print(z)
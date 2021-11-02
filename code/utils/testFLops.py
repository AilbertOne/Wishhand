import torchvision.models as models
import torch
from flops_counter import get_model_complexity_info
# from pthflops import get_model_complexity_info
net = models.vgg16()
flops, params = get_model_complexity_info(net,(224,224),as_strings=True,print_per_layer_stat=True)
print("Flops: {}".format(flops))
print("params:" + params)





#
#
# with torch.cuda.device(0):
#   net = models.resnet18()
#   flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, 		                        print_per_layer_stat=True) #不用写batch_size大小，默认batch_size=1
#   print('Flops:  ' + flops)
#   print('Params: ' + params)

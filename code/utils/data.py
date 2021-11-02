import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
transform1 = {
        "img": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    # transforms.ToTensor(),
                                    transforms.Normalize([0.545, 0.482, 0.437], [0.176, 0.193, 0.207])
                                     ]),
        "tensor": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                    transforms.Normalize([0.545, 0.482, 0.437], [0.176, 0.193, 0.207])
                                   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}
def tensor_to_np(tensor):

    img = tensor.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def lapras(img):
    g0 = cv2.pyrDown(img)
    l0 = img - cv2.pyrUp(g0)
    return l0

def np_to_pil(img1):
    return Image.fromarray(np.uint8(img1))

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().unsqueeze(0)
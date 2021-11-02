import os
import json
import time

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# from model.MobResNet import MobResNet18
from models.MobResNet import MobResNet18

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.545, 0.482, 0.438], [0.176, 0.193, 0.208])])
         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "./images/none892.jpg"
    # print(img_path)
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # print(img)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = MobResNet18(num_classes=8).to(device)

    # load model weights
    weights_path = "./weightsnew/MobResNet18.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))


    # prediction
    model.eval()
    start = time.process_time()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        end = time.process_time()
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())

    print('Running time: %s Seconds'%(end-start))
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()

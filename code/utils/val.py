import os
import json

import torch
from PIL import Image
from torchvision import transforms

# from model import efficientnet_b0 as create_model
from models.MobCBAM_ResNet import MobResNet18
# from models.vggnet import vgg #------------
# a = open(r'D:\pycharmFiles\pytorch_classification\Test9_efficientNet\pre_results.txt','a')

def single_img_pre(model, img_path):
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.Resize((448, 224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.545, 0.482, 0.438], [0.176, 0.193, 0.208])
        ])


    # img_files = "D:/pycharmFiles/TensorFlow2.0_ResNet-master/original_data/img/datasets/test/"

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # read class_indict
    json_path = '../class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 开始预测
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    prdict_acc = predict[predict_cla].numpy()
    print(print_res)

    return class_indict[str(predict_cla)], prdict_acc



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    # model = MobileNetV2(num_classes=8).to(device)
    model = MobResNet18(num_classes=8).to(device) #-------------
    # load model weights
    model_weight_path = "../weightsnew/mobcbam_res.pth" #--------------
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # load test_dir
    test_imgs_dir =r"D:\learnfile\deep-learning-for-image-processing-master\data_set\wishhand_data\test"
    acc_info_txt="../txtnew/mobcbam_res.txt" #----------------
    with open(acc_info_txt,"w") as file:
        for root, dirs,files in os.walk(test_imgs_dir):
            for sub_dir in dirs:
                file.writelines(["预测类别信息：", sub_dir,"\n"])
                test_acc = 0
                sub_dir_path = os.path.join(root, sub_dir)
                imgs = [img for img in os.listdir(sub_dir_path)]
                imgs_class_nums= len(imgs)
                print(imgs_class_nums)
                for img in imgs:
                    sing_img_path = os.path.join(sub_dir_path, img)
                    test_class, pre_acc=single_img_pre(model, sing_img_path)
                    # round（），对指定小数位进行四舍五入
                    file.writelines([sing_img_path,":     ", test_class,": ", str(round(pre_acc.item(), 4)),"\n"])
                    if test_class==sub_dir:
                        test_acc += 1
                pre_class_acc = test_acc / imgs_class_nums
                print_res = "class info: {}   aver_prob: {:.3}".format(sub_dir, pre_class_acc)
                file.writelines(["aver_prob of ",sub_dir,":",str(round(pre_class_acc, 4)),"\n"])
                print(print_res)
    file.close()





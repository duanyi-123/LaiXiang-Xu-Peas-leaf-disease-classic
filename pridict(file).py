import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model_DCN_self_AA import mobilenetv3_small


def predict_image(model, data_transform, img_path, class_indict, device):
    """预测单张图片"""
    # load image
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    plt.show()

    return predict_cla, predict


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load image folder
    img_folder = r"D:\深度学习（豌豆）\mobilenetV-3\原图像\healthy"
    assert os.path.exists(img_folder), "folder: '{}' dose not exist.".format(img_folder)

    # 获取文件夹中所有图片文件
    img_files = [os.path.join(img_folder, f) for f in os.listdir(img_folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # read class_indict
    json_path = r"D:\深度学习（豌豆）\mobilenetV-3\pea_leaves_class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = mobilenetv3_small(num_classes=5).to(device)
    # load model weights
    model_weight_path = r"D:\深度学习（豌豆）\mobilenetV-3\mobilenetv3-small-55df8e1f.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 预测所有图片
    for img_path in img_files:
        print(f"\nProcessing image: {os.path.basename(img_path)}")
        predict_image(model, data_transform, img_path, class_indict, device)


if __name__ == '__main__':
    main()
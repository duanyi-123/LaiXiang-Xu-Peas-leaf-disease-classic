import os
import json
from PIL import Image
from torchvision import transforms
import torch
from model_DCN_self_AA import mobilenetv3_small


def predict_image(model, data_transform, img_path, device):
    """预测单张图片并返回预测结果"""
    # load image
    img = Image.open(img_path).convert('RGB')
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    return predict_cla


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 根文件夹路径，包含多个类别子文件夹
    root_folder = r"D:\深度学习（豌豆）\mobilenetV-3\saved_datasets\test"
    assert os.path.exists(root_folder), "folder: '{}' dose not exist.".format(root_folder)

    # 获取所有类别子文件夹
    class_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    # 创建类别到索引的映射
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}

    # 保存类别索引到JSON文件
    with open("class_indices.json", "w") as f:
        json.dump(class_to_idx, f)

    # read class_indict
    json_path = "class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 将类别字典转换为列表
    class_names = list(class_indict.keys())

    # create model
    model = mobilenetv3_small(num_classes=len(class_folders)).to(device)
    # load model weights
    model_weight_path = r"D:\深度学习（豌豆）\mobilenetV-3\mobilenetv3-small-55df8e1f.pth(zuizhong)"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 初始化真实标签和预测标签列表
    true_labels = []
    pred_labels = []

    # 遍历每个类别文件夹
    for class_name in class_folders:
        class_folder = os.path.join(root_folder, class_name)
        # 获取文件夹中所有图片文件
        img_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # 预测所有图片
        for img_path in img_files:
            # 从文件夹名称获取真实标签
            true_label_idx = class_to_idx[class_name]
            true_labels.append(true_label_idx)

            # 预测图片
            pred_label = predict_image(model, data_transform, img_path, device)
            pred_labels.append(pred_label)

            print(f"Image: {os.path.basename(img_path)}")
            print(f"True: {class_name}, Predicted: {class_names[pred_label]}")


if __name__ == '__main__':
    main()
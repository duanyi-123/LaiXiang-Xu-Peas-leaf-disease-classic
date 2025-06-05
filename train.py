import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from model_original import (mobilenetv3_small)
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 数据集路径
image_path = r"D:\深度学习（豌豆）\mobilenetV-3\date"

# 加载完整数据集
full_dataset = datasets.ImageFolder(root=image_path, transform=data_transform["train"])

# 获取数据集大小
dataset_size = len(full_dataset)

# 划分数据集
train_size = int(0.6 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# 获取分类的名称
flower_list = full_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())

# 创建数据加载器
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 定义模型
net = mobilenetv3_small(num_classes=5).to(device)
model_weight_path = r"D:\深度学习（豌豆）\mobilenetV-3\mobilenetv3-small-55df8e1f.pth"

# 载入模型权重
pre_weights = torch.load(model_weight_path)
pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

# 冻结除最后几层以外的权重
for param in list(net.features.parameters())[-10:]:
    param.requires_grad = True

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002, weight_decay=1e-5)

# 设置存储权重路径
save_path = r"D:\深度学习（豌豆）\mobilenetV-3\mobilenetv3-small-55df8e1f.pth"
best_acc = 0.0

# 创建保存预测结果的文件夹
predictions_dir = r"D:\深度学习（豌豆）\mobilenetV-3\predictions"
os.makedirs(predictions_dir, exist_ok=True)

# 训练循环
for epoch in range(50):
    # train
    net.train()
    running_loss = 0.0
    running_corrects = 0

    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    train_loss = running_loss / len(train_loader)
    train_acc = running_corrects.double() / len(train_dataset)

    # validate
    net.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for data_test in val_loader:
            test_images, test_labels = data_test
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            outputs = net(test_images)
            loss = loss_function(outputs, test_labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == test_labels.data)

    val_loss = val_loss / len(val_loader)
    val_acc = val_corrects.double() / len(val_dataset)

    print("[epoch %d] train_loss: %.3f  val_loss: %.3f  train_acc: %.3f  val_acc: %.3f" %
          (epoch + 1, train_loss, val_loss, train_acc, val_acc))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), save_path)

# 测试模型并保存预测结果
net.eval()
test_acc = 0.0
with torch.no_grad():
    for batch_idx, data_test in enumerate(test_loader):
        test_images, test_labels = data_test
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        outputs = net(test_images)
        predict_y = torch.max(outputs, dim=1)[1]
        test_acc += (predict_y == test_labels).sum().item()

        # 保存预测结果
        for i in range(len(test_images)):
            # 获取原始图像路径
            img_idx = batch_idx * batch_size + i
            if img_idx >= len(test_dataset):  # 防止超出范围
                continue

            # 获取原始图像路径
            img_path = test_dataset.dataset.samples[test_dataset.indices[img_idx]][0]

            # 创建目标路径
            img_name = os.path.basename(img_path)
            true_label = cla_dict[test_labels[i].item()]
            pred_label = cla_dict[predict_y[i].item()]

            # 创建子文件夹：真实类别_预测类别
            save_subdir = os.path.join(predictions_dir, f"{true_label}_{pred_label}")
            os.makedirs(save_subdir, exist_ok=True)

            # 保存图像
            save_path = os.path.join(save_subdir, img_name)

            # 复制原始图像到预测文件夹
            shutil.copy2(img_path, save_path)

            # 或者保存处理后的图像（如果需要）
            # img = transforms.ToPILImage()(test_images[i].cpu())
            # img.save(save_path)

    print("Test Accuracy: %.3f" % (test_acc / len(test_dataset)))

print("Finished Training")
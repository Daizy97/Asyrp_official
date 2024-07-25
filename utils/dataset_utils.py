import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd

# 假设图像文件名和属性文件中的对应
class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, labels, transform=None):
        self.img_folder = img_folder
        self.labels = labels
        self.transform = transform
        self.img_names = labels.index

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_folder, self.img_names[idx])
        image = Image.open(img_name)
        label = self.labels.loc[self.img_names[idx], 'Male']
        if self.transform:
            image = self.transform(image)
        return image, label

def get_celeba_dataset(dataset_path, batch_size=32, image_size=256):
    # 定义转换操作，你可以根据需要添加更多的数据增强
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ])

    # 加载数据集
    # 假设 data_dir 是你的数据集路径，标签在属性文件中
    data_dir = os.path.join(dataset_path, 'celeba-hq', 'celeba-256')
    attributes_path = os.path.join(dataset_path, 'list_attr_celeba.txt')

    # 解析属性文件，得到性别标签
    attributes = pd.read_csv(attributes_path, delim_whitespace=True, header=1)  # -1：Female, 1：Male
    attributes['Male'] = (attributes['Male'] + 1) // 2  # 将 Male 转换为二分类标签

    # 删选出存在于CelebA-HQ文件夹中的图片及其对应的性别标签
    img_names = os.listdir(data_dir)
    filtered_attributes = attributes.loc[img_names]

    # 创建数据集
    dataset = CelebADataset(data_dir, filtered_attributes, transform)

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_afhq_dataset(dataset_path, batch_size=32, image_size=256):
    # 定义转换操作
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整图片大小
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ])

    # 加载数据集
    train_data = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
''' train target classifier'''
import os
import sys
import numpy as np
import argparse
import logging
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import json
import requests

IMAGE_SIZE = 256
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'eval'])
parser.add_argument('-d', '--dataset', type=str, default='ImageNet', choices=['CelebA_HQ', 'AFHQ', 'ImageNet'])
parser.add_argument('-m', '--model', type=str, default='resnet50')
parser.add_argument('--pretrained', action='store_true', help='if use pretrained model')
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-e', '--epochs', type=int, default=90)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--save_model', action='store_false')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import logs
from configs.paths_config import DATASET_PATHS
# from datasets.data_utils import get_dataset, get_dataloader
from utils.dataset_utils import get_celeba_dataset, get_afhq_dataset

# add device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info("Using device: {}".format(device))

# set logging
timestamp = str(datetime.now())[:-7]
filename = timestamp + ' ' + args.mode + '-' + args.dataset + '-' + args.model
savedir = os.path.join(os.getcwd(), 'classifiers', filename)
if not os.path.exists(savedir):
    os.makedirs(savedir)
hps_str = '{} mode={} dataset={} model={} pretrained={} batch size={} learning rate={} epoch={}'\
    .format(timestamp, args.mode, args.dataset, args.model, args.pretrained, args.batch_size, args.learning_rate, args.epochs)
log_path = '{}/{}.log'.format(savedir, hps_str)
log = logs.Logger(log_path)
log.print('\nAll hps: {}'.format(hps_str))

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.benchmark = True

# get model
if args.dataset == 'CelebA_HQ':
    num_classes = 2  # 2个类别：男性和女性
    if args.model == 'resnet50':
        IMAGE_SIZE = 224
        if args.pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)
    # 修改最后的全连接层，以适应二分类任务
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
elif args.dataset == 'AFHQ':
    num_classes = 3
    if args.model == 'resnet50':
        IMAGE_SIZE = 224
        if args.pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
elif args.dataset == 'ImageNet':
    num_classes = 1000
    if args.model == 'resnet50':
        IMAGE_SIZE = 224
        if args.pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
else:
    print("Invalid dataset!")
    sys.exit(0)

def evaluate_model(args, model, test_loader, epoch):
    model.eval()  # Set model to evaluate mode
    corrects = 0
    total = 0
    time_s = datetime.now()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (preds == labels).sum().item()

    accuracy = corrects / total
    time_e = datetime.now()
    time = (time_e - time_s).seconds
    log.print(f'Epoch {epoch}/{args.epochs - 1}, Accuracy: {accuracy} ({corrects}/{total}), time: {time}s')


def train(args):
    # get datasets
    # train_dataset, test_dataset = get_dataset(args.dataset, DATASET_PATHS, config)
    if args.dataset == 'CelebA_HQ':
        train_loader, test_loader = get_celeba_dataset(DATASET_PATHS[args.dataset], args.batch_size, IMAGE_SIZE)
    elif args.dataset == 'AFHQ':
        train_loader, test_loader = get_afhq_dataset(DATASET_PATHS[args.dataset], args.batch_size, IMAGE_SIZE)
    elif args.datest == 'ImageNet':  # 不用训练ImageNet
        print("No need to train ImageNet!")
        sys.exit(0)
    else:
        print("Invalid dataset!")
        sys.exit(0)

    # 定义损失函数和优化器
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    # 训练模型
    print('Start training...')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs - 1}')
        time_s = datetime.now()
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=(running_loss / (i + 1)))

        epoch_loss = running_loss / len(train_loader.dataset)
        time_e = datetime.now()
        time = (time_e - time_s).seconds
        log.print(f'Epoch {epoch}/{args.epochs - 1}, Loss: {epoch_loss}, time: {time}s')
        # 调用评估函数
        evaluate_model(args, model, test_loader, epoch)
        # 保存模型
        torch.save(model.state_dict(), os.path.join(savedir, args.model + f"epoch{epoch}.pth"))

def inference(args, img_paths, model_path):
    # 下载 ImageNet 类别标签
    if args.dataset == 'ImageNet':
        # LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        # response = requests.get(LABELS_URL)
        # imagenet_labels = response.json()
        json_path = os.path.join(os.getcwd(), 'classifiers', 'imagenet-simple-labels.json')
        with open(json_path, 'r') as f:
            imagenet_labels = json.load(f)

    # 加载模型
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 预处理图像
    if args.model == 'resnet50':
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        print("Invalid model!")
        sys.exit(0)
    if os.path.isfile(img_paths):  # 如果路径是图片
        img = Image.open(img_paths)
        img = transform(img).unsqueeze(0).to(device)
        # 预测
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
            pred = pred.item()
            if args.dataset == 'CelebA_HQ':
                label = 'Female' if pred == 0 else 'Male'
            elif args.dataset == 'AFHQ':
                label = 'Cat' if pred == 0 else 'Dog' if pred == 1 else 'Wildlife'
            elif args.dataset == 'ImageNet':
                label = imagenet_labels[pred]
            print(f'Predicted index: {pred}, label: {label}.')
    else:
        img_names = os.listdir(img_paths)
        num_imgs = len(img_names)
        if args.dataset == 'CelebA_HQ':
            num_male, num_female = 0, 0
            for img_name in img_names:
                img = Image.open(os.path.join(img_paths, img_name))
                img = transform(img).unsqueeze(0).to(device)
                output = model(img)
                _, pred = torch.max(output, 1)
                if pred.item() == 0:
                    num_female += 1
                else:
                    num_male += 1
            print(f'Female accuracy: {num_female / num_imgs} ({num_female}/{num_imgs})')
            print(f'Male accuracy: {num_male / num_imgs} ({num_male}/{num_imgs})')
        elif args.dataset == 'AFHQ':
            num_cat, num_dog, num_wild = 0, 0, 0
            for img_name in img_names:
                img = Image.open(os.path.join(img_paths, img_name))
                img = transform(img).unsqueeze(0).to(device)
                output = model(img)
                _, pred = torch.max(output, 1)
                if pred.item() == 0:
                    num_cat += 1
                elif pred.item() == 1:
                    num_dog += 1
                else:
                    num_wild += 1
            print(f'Cat accuracy: {num_cat / num_imgs} ({num_cat}/{num_imgs})')
            print(f'Dog accuracy: {num_dog / num_imgs} ({num_dog}/{num_imgs})')
            print(f'Wildlife accuracy: {num_wild / num_imgs} ({num_wild}/{num_imgs})')
        elif args.dataset == 'ImageNet':
            num_correct = 0
            for img_name in img_names:
                img = Image.open(os.path.join(img_paths, img_name))
                img = transform(img).unsqueeze(0).to(device)
                output = model(img)
                _, pred = torch.max(output, 1)
                # TODO: 所在文件夹的图片应为同一类别，类别序号在文件夹名称中显示
                if pred.item() == int(img_name.split('_')[-1].split('.')[0]):
                    num_correct += 1
            print(f'Accuracy: {num_correct / num_imgs} ({num_correct}/{num_imgs})')

if __name__ == '__main__':
    if args.mode == 'train':
        time_s = datetime.now()
        train(args)
        time_e = datetime.now()
        time = (time_e - time_s).seconds
        hours, remainder = divmod(time, 3600)
        minutes, seconds = divmod(remainder, 60)
        log.print('Total time: {} hours {} minutes {} seconds'.format(hours, minutes, seconds))

    elif args.mode == 'test':  # 测试单张图片
        # evaluate_model(args, model, test_loader, epoch=0)
        # test(args)
        if args.dataset == 'CelebA_HQ':
            image_path = os.path.join(DATASET_PATHS[args.dataset], 'celeba-hq', 'celeba-256', '000281.jpg')
            model_path = os.path.join(os.getcwd(), 'classifiers',
                                      '2023-11-11 16:38:59 train-CelebA_HQ-resnet50', 'resnet50epoch89.pth')
        elif args.dataset == 'AFHQ':
            image_path = os.path.join(DATASET_PATHS[args.dataset], 'val', 'wild', 'flickr_wild_000129.jpg')
            model_path = os.path.join(os.getcwd(), 'classifiers',
                                  '2023-11-11 13:15:52 train-AFHQ-resnet50', 'resnet50epoch89.pth')
        elif args.dataset == 'ImageNet':
            image_path = os.path.join(DATASET_PATHS[args.dataset], 'n01440764', 'ILSVRC2012_val_00002138.JPEG')
            model_path = os.path.join(os.getcwd(), 'classifiers', 'imagenet1k_resnet50_v1.pth')
        else:
            print("Invalid dataset!")
            sys.exit(0)
        print(f'Using model {model_path}...')
        print(f'Inferencing {image_path}...')
        inference(args, image_path, model_path)

    elif args.mode == 'eval':  # 测试一个目录下的所有图片
        if args.dataset == 'CelebA_HQ':
            image_path = '/data/daizeyu/Asyrp/runs/2023-11-14_07:48:47_celeba_hq_man_CUSTOM_t999_ninv40_ngen40/test_images/edited'
            # image_path = '/data/daizeyu/Asyrp/runs/2023-11-14_07:48:47_celeba_hq_man_CUSTOM_t999_ninv40_ngen40/image_samples/reconstructed'
            model_path = os.path.join(os.getcwd(), 'classifiers', 'celeba-hq_resnet50_gender.pth')
        elif args.dataset == 'AFHQ':
            image_path = ''
            model_path = os.path.join(os.getcwd(), 'classifiers', 'afhq_resnet50.pth')
        elif args.dataset == 'ImageNet':
            image_path = ''
            model_path = os.path.join(os.getcwd(), 'classifiers', 'imagenet1k_resnet50_v1.pth')
        else:
            print("Invalid dataset!")
            sys.exit(0)
        print(f'Using model {model_path}...')
        print(f'Inferencing {image_path}...')
        inference(args, image_path, model_path)
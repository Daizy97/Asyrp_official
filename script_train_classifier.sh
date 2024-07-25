#!/usr/bin/zsh
python train_classifier.py --gpu '6' -d 'CelebA_HQ' --mode 'train' -e 90 --pretrained & sleep 1
python train_classifier.py --gpu '7' -d 'CelebA_HQ' --mode 'train' -e 90

#python train_classifier.py --gpu '4' -d 'AFHQ' --mode 'train' -e 90 --pretrained & sleep 1
#python train_classifier.py --gpu '5' -d 'AFHQ' --mode 'train' -e 90
# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm
from model import Model


def main():
    # set device on GPU if available, else CPU
    
    #device = 'cpu'
    device = torch.device("cuda:0")

    n = 8 #batch_size
    epoch = 10000
    interval = 500 #观测、保存模型的间隔

    train_content_dir, train_style_dir = 'content', 'style'
    test_content_dir, test_style_dir = 'content', 'style'
    # prepare dataset and dataLoader
    train_dataset = PreprocessDataset(train_content_dir, train_style_dir)
    test_dataset = PreprocessDataset(test_content_dir, test_style_dir)
    iters = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size = n, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = n, shuffle = False)
    test_iter = iter(test_loader)

    # set model and optimizer
    model = Model().to(device)
    #if reuse is not None:
    model.load_state_dict(torch.load('last_model_state.pth', map_location=lambda storage, loc: storage))
    optimizer = Adam(model.parameters(), lr=5e-5)

    # start training
    loss_list = []
    for e in range(1, epoch + 1):
        for i, (content, style) in tqdm(enumerate(train_loader, 1)):
            content = content.to(device)
            style = style.to(device)
            loss = model(content, style)

            optimizer.zero_grad() #梯度零化
            loss.backward() #反向传播，算新的梯度
            optimizer.step() #更新参数

            loss_list.append(loss.item())
            print(loss.item())

            if i % interval == 0: #保存模型
                torch.save(model.state_dict(), 'model_state.pth')

if __name__ == '__main__':
    main()

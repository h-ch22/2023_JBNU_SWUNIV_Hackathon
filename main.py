import copy
import os
import time

import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import os

from PIL import Image
from torch import Tensor, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn import datasets, model_selection
from CoverCNN import CoverCNN
from EfficientNet import EfficientNet
from torch.optim.lr_scheduler import ReduceLROnPlateau

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b


# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data
    return loss, metric

def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights=params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr= {}'.format(epoch, num_epochs - 1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights!')

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' % (
        train_loss, val_loss, 100 * val_metric, (time.time() - start) / 60))
        print('-' * 10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


if __name__ == '__main__':
    CATEGORIES = {0: 'ART', 1: 'BUSINESS', 2: 'CHILD', 3: 'COMIC', 4: 'ELESCHOOL_REF', 5: 'ESSAY', 6: 'EXAMINATION', 7: 'FOREIGN_LANGUAGE',
                  8: 'HEALTH', 9: 'HISTORY', 10: 'HOBBY', 11: 'HOME', 12: 'HUMANITIES', 13: 'IT', 14: 'JUNIOR', 15: 'MAGAZINE', 16: 'MID_HIGHSCHOOL_REF',
                  17: 'NOVEL', 18: 'RELIGION', 19: 'SCIENCE', 20: 'SELF-DEVELOPMENT', 21: 'SOCIAL', 22: 'TECHNICAL',
                  23: 'TEENAGER', 24: 'TRIP'}

    # TRAIN_PATH = 'C:/Users/USER/Desktop/2023/JBNU_SWUNIV_HACKATHON/DATA/TRAIN/'
    TRAIN_PATH = '/Users/changjinha/Desktop/2023/JBNU_SWUNIV_HACKATHON/DATA/TRAIN/'

    data = []
    label = []
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    for i, d in enumerate(CATEGORIES.values()):
        files = os.listdir(TRAIN_PATH + d)

        for f in files:
            if '.jpg' in f:
                img = Image.open(TRAIN_PATH + d + '/' + f, 'r')
                resize_img = img.resize((224, 224))

                r, g, b = resize_img.split()
                r_resize_img = np.asarray(np.float32(r) / 255.0)
                b_resize_img = np.asarray(np.float32(g) / 255.0)
                g_resize_img = np.asarray(np.float32(b) / 255.0)

                rgb_resize_img = np.asarray([r_resize_img, b_resize_img, g_resize_img])

                data.append(rgb_resize_img)
                label.append(i)

    data = np.array(data, dtype='float32')
    label = np.array(label, dtype='int64')

    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data, label, test_size=0.1)

    train_X = torch.from_numpy(train_X).float()
    train_Y = torch.from_numpy(train_Y).long()

    test_X = torch.from_numpy(test_X).float()
    test_Y = torch.from_numpy(test_Y).long()

    train = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train, batch_size=80, shuffle=True)

    test = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test, batch_size=80, shuffle=True)

    print(train_X.shape)

    efficientNet = EfficientNet()
    model = efficientNet.efficientnet_b0().to(device)

    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=0.01)

    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

    params_train = {
        'num_epochs': 100,
        'optimizer': opt,
        'loss_func': loss_func,
        'train_dl': train_loader,
        'val_dl': test_loader,
        'sanity_check': False,
        'lr_scheduler': lr_scheduler,
        'path2weights': './models/weights.pt',
    }
    createFolder('./models')
    model, loss_hist, metric_hist = train_val(model, params_train)


    # learning_rate = 0.001
    # model = CoverCNN()
    # efficientNet = EfficientNet()
    # model = efficientNet.efficientnet_b0().to(device)
    #
    # model.to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # num_epochs = 200
    # count = 0
    # loss_list = []
    # iteration_list = []
    # accuracy_list = []
    #
    # predictions_list = []
    # labels_list = []
    #
    # for epoch in range(num_epochs):
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         train = Tensor(images.view(-1, 3, 128, 128))
    #         labels = Tensor(labels)
    #         outputs = model(train)
    #         loss = criterion(outputs, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         count += 1
    #
    #         if not (count % 50):
    #             total = 0
    #             correct = 0
    #
    #             for images, labels in test_loader:
    #                 images, labels = images.to(device), labels.to(device)
    #                 labels_list.append(labels)
    #                 test = Tensor(images.view(-1, 3, 128, 128))
    #                 outputs = model(test)
    #
    #                 predictions = torch.max(outputs, 1)[1].to(device)
    #                 predictions_list.append(predictions)
    #                 correct += (predictions == labels).sum()
    #                 total += len(labels)
    #
    #                 accuracy = correct * 100 / total
    #                 loss_list.append(loss.data)
    #                 iteration_list.append(count)
    #                 accuracy_list.append(accuracy)
    #
    #         if not (count % 500):
    #             print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
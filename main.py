import os
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

from PIL import Image
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn import datasets, model_selection
from CoverCNN import CoverCNN
from EfficientNet import EfficientNet

if __name__ == '__main__':
    CATEGORIES = {0: 'ART', 1: 'BUSINESS', 2: 'CHILD', 3: 'COMIC', 4: 'ELESCHOOL_REF', 5: 'ESSAY', 6: 'EXAMINATION', 7: 'FOREIGN_LANGUAGE',
                  8: 'HEALTH', 9: 'HISTORY', 10: 'HOBBY', 11: 'HOME', 12: 'HUMANITIES', 13: 'IT', 14: 'JUNIOR', 15: 'MAGAZINE', 16: 'MID_HIGHSCHOOL_REF',
                  17: 'NOVEL', 18: 'RELIGION', 19: 'SCIENCE', 20: 'SELF-DEVELOPMENT', 21: 'SOCIAL', 22: 'TECHNICAL',
                  23: 'TEENAGER', 24: 'TRIP'}

    TRAIN_PATH = 'C:/Users/USER/Desktop/2023/JBNU_SWUNIV_HACKATHON/DATA/TRAIN/'

    data = []
    label = []
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    for i, d in enumerate(CATEGORIES.values()):
        files = os.listdir(TRAIN_PATH + d)

        for f in files:
            if '.jpg' in f:
                img = Image.open(TRAIN_PATH + d + '/' + f, 'r')
                resize_img = img.resize((128, 128))

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

    learning_rate = 0.001
    # model = CoverCNN()
    efficientNet = EfficientNet()
    model = efficientNet.efficientnet_b0().to(device)
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 200
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []

    predictions_list = []
    labels_list = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            train = Tensor(images.view(-1, 3, 128, 128))
            labels = Tensor(labels)
            outputs = model(train)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

            if not (count % 50):
                total = 0
                correct = 0

                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)
                    test = Tensor(images.view(-1, 3, 128, 128))
                    outputs = model(test)

                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()
                    total += len(labels)

                    accuracy = correct * 100 / total
                    loss_list.append(loss.data)
                    iteration_list.append(count)
                    accuracy_list.append(accuracy)

            if not (count % 500):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
import pickle
import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
from utils import *
import time


class Baseline(nn.Module):
    def __init__(self, width, sd, dropout, init_type, depth=1):
        super(Baseline, self).__init__()
        self.depth = depth
        self.linear1 = nn.Linear(32 * 32 * 3, width)
        if depth > 1:
            self.linear2 = nn.Linear(width, width)
            self.linear3 = nn.Linear(width, width)
        if depth > 3:
            self.linear4 = nn.Linear(width, width)
        if depth > 4:
            self.linear5 = nn.Linear(width, width)
            self.linear6 = nn.Linear(width, width)
            self.linear7 = nn.Linear(width, width)
            self.linear8 = nn.Linear(width, width)
            self.linear9 = nn.Linear(width, width)
            self.linear10 = nn.Linear(width, width)
        self.relu = nn.ReLU()
        self.num_layers = 2
        self.dropout = nn.Dropout(dropout)
        if init_type == "xavier":
            torch.nn.init.xavier_uniform_(self.linear1.weight)
            if depth > 1:
                torch.nn.init.xavier_uniform_(self.linear2.weight)
                torch.nn.init.xavier_uniform_(self.linear3.weight)
            if depth>3:
                torch.nn.init.xavier_uniform_(self.linear4.weight)
            if depth > 4:
                torch.nn.init.xavier_uniform_(self.linear5.weight)
                torch.nn.init.xavier_uniform_(self.linear6.weight)
                torch.nn.init.xavier_uniform_(self.linear7.weight)
                torch.nn.init.xavier_uniform_(self.linear8.weight)
                torch.nn.init.xavier_uniform_(self.linear9.weight)
                torch.nn.init.xavier_uniform_(self.linear10.weight)
        elif init_type == "normal":
            self.linear1.weight.data.normal_(0.0, sd)
            if depth > 1:
                self.linear2.weight.data.normal_(0.0, sd)
                self.linear3.weight.data.normal_(0.0, sd)
            if depth>3:
                self.linear4.weight.data.normal_(0.0, sd)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        if self.depth > 1:
            x = self.linear2(x)
            x = self.relu(x)
            x = self.linear3(x)
            x = self.relu(x)
        if self.depth > 3:
            x = self.linear4(x)
            x = self.relu(x)
        if self.depth > 4:
            x = self.linear5(x)
            x = self.relu(x)
            x = self.linear6(x)
            x = self.relu(x)
            x = self.linear7(x)
            x = self.relu(x)
            x = self.linear8(x)
            x = self.relu(x)
            x = self.linear9(x)
            x = self.relu(x)
            x = self.linear10(x)
            x = self.dropout(x)
            x = self.relu(x)
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name, preform_whitening):
        self.images, self.labels = self.load_images(file_name, preform_whitening)

    def load_images(self, file_name, preform_whitening=False):
        with open(file_name, "rb") as df:
            loaded = pickle.load(df)
            data = loaded["data"]
            if preform_whitening:
                pca = PCA(whiten=True)
                pca.fit(data)
                data = pca.transform(data)
            labels = loaded["labels"]
            return tensor(data), tensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def train_baseline(width, lr, momentum, sd, wd, dropout, optimizer_type, init_type, preform_whitening,name="name",depth=1):
    batch_size = 1024

    device = get_training_device()
    model = Baseline(width, sd, dropout, init_type,depth=depth).to(device)

    train_set = Dataset("../data/train.pkl", preform_whitening=preform_whitening)
    test_set = Dataset("../data/test.pkl", preform_whitening=False)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    # Train the model
    acc_test = []
    acc_train = []
    loss_train = []
    loss_test = []

    num_epoch = 80
    for epoch in range(num_epoch):
        cur_loss_train = 0
        cur_loss_test = 0
        cur_test_acc = 0
        cur_train_acc = 0

        total_train_samples = 0
        total_test_samples = 0

        # train
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            images = images.float()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            cur_loss_train += loss.item()

            # Backprop and perform SGD optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            _, prediction = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (prediction == labels).sum().item()
            cur_train_acc = correct / total

            total_train_samples += labels.size(0)
        cur_loss_train /= (len(train_loader) + 1)

        # eval on test
        with torch.no_grad():
            model.eval()
            for j, (images, labels) in enumerate(test_loader):
                images = images.float()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, prediction = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                cur_loss_test += loss.item()

                total = labels.size(0)
                correct = (prediction == labels).sum().item()
                cur_test_acc = correct / total

                total_test_samples += labels.size(0)

            cur_loss_test /= (len(test_loader) + 1)

        loss_train.append(cur_loss_train)
        acc_train.append(cur_train_acc)
        loss_test.append(cur_loss_test)
        acc_test.append(cur_test_acc)

        max_train_acc_index=0
        if epoch == num_epoch-1:
            max_train_acc_index = acc_train.index(max(acc_train))
            print("Best epoch {}, Train Accuracy: {}%, test accuracy: {}% , TrainLoss: {} , Testloss: {}".format(
                max_train_acc_index,
                round(acc_train[max_train_acc_index] * 100,2),
                round(acc_test[max_train_acc_index] * 100,2),
                round(loss_train[max_train_acc_index],2),
                round(loss_test[max_train_acc_index],2)
            ))

    torch.save(model.state_dict(), "./models/q2_base.pt")
    plot_res(range(num_epoch), loss_train, "train loss", name+" loss plot", "epoch", "loss", "Loss vs Epoch", Y2=loss_test, label2="test loss")
    plot_res(range(num_epoch), acc_train, "train accuracy", name+" accuray plot", "epoch", "accuracy", "Accuracy vs Epoch", Y2=acc_test, label2="test accuracy")



    return acc_train[max_train_acc_index], acc_test[max_train_acc_index], loss_train[max_train_acc_index], loss_test[max_train_acc_index]

# Q2 - baseline grid search
# for lr in [0.1, 0.01, 0.001]:
#     for m in [0.9, 0.5, 0.1]:
#         for sd in [1.0, 0.5, 0.1]:
#             print(f"running with lr:{lr}, m:{m}, sd:{sd}")
#             train_baseline(lr, m, sd)
# # best configuration: lr=0.01, momentum=0.9 sd=0.1,


# # compare optimizers:
# print("Adam: lr=0.01, momentum=0.9 sd=0.1")
# train_baseline(256 ,0.01, 0.9, 0.1,False,0,"adam","normal",False, name="Adam Optimizer")
# print("SGD: lr=0.01, momentum=0.9 sd=0.1")
# train_baseline(256 ,0.01, 0.9, 0.1,False,0,"sgd","normal",False, name="SGD Optimizer")
# # best optimizer is SGD


# # compare init values:
# print("Xavier: SGD lr=0.01, momentum=0.9 sd=0.1")
# train_baseline(256 ,0.01, 0.9, 0.1,False,0,"sgd","Xavier",False, name="Xavier Init")
# print("Normal init: SGD lr=0.01, momentum=0.9 sd=0.1")
# train_baseline(256 ,0.01, 0.9, 0.1,False,0,"sgd","normal",False, name="Normal Init")
# # best init is Xavier


# # compare regularization:
# for dropout in [0, 0.001, 0.005, 0.009]:
#     for wd in [0, 0.001, 0.005, 0.009]:
#         print(f"Dropout={dropout}, Weight Decay={wd}:\n Xavier, SGD, lr=0.01, momentum=0.9 sd=0.1")
#         start = time.time()
#         train_baseline(256, 0.01, 0.9, 0.1, wd, dropout, "sgd", "normal", False, name=" Init")
#         end = time.time()
#         print(f"runtime: {end - start} \n")
#train_baseline(256, 0.01, 0.9, 0.1, 0, 0.005, "sgd", "normal", False, name="Regularization")


# # Preprocessing:
# start=time.time()
# train_baseline(256, 0.01, 0.9, 0.1, 0, 0.005, "sgd", "normal",True, name="Preprocessing")
# end = time.time()
# print(f"runtime: {end - start} \n")


# # Width:
# best dropout=0.005, wd=0, comparing width
# acc_train_by_width = []
# loss_train_by_width = []
# acc_test_by_width = []
# loss_test_by_width = []
# for i in [6,10,12]:
#     width = 2 ** i
#     print(f"width {width}: Xavier, SGD, lr=0.01, momentum=0.9 sd=0.1, dropout=0.005, wd=0")
#     start = time.time()
#     acc_train, acc_test, loss_train, loss_test = train_baseline(width, 0.01, 0.9, 0.1, 0, 0.005, "sgd", "xavier", preform_whitening=False)
#     end = time.time()
#     acc_train_by_width.append(acc_train)
#     loss_train_by_width.append(loss_train)
#     acc_test_by_width.append(acc_test)
#     loss_test_by_width.append(loss_test)
#     print(f"runtime: {end - start} \n")
#
# print(f"loss_train_by_width: {loss_train_by_width} \nloss_test_by_width: {loss_test_by_width} \nacc_train_by_width:"
#       f" {acc_train_by_width} \nacc_test_by_width: {acc_test_by_width}")
#
# plot_res([64, 1024, 4096], acc_train_by_width, 'train accuracy', "q2_acc_vs_width", "Width", "Accuracy"
#          , "Accuracy vs Width", Y2=acc_test_by_width , label2='test accuracy')
#
# plot_res([64, 1024, 4096], loss_train_by_width, 'train loss', "q2_loss_vs_width", "Width", "Loss"
#          , "Loss vs Width", Y2=loss_test_by_width , label2='test loss')


# # Depth:
# acc_train_by_depth = []
# loss_train_by_depth = []
# acc_test_by_depth = []
# loss_test_by_depth = []
# for depth in [3, 4, 10]:
#     print(f"depth {depth}: Xavier, SGD, lr=0.01, momentum=0.9 sd=0.1, dropout=0.005, wd=0")
#     start = time.time()
#     acc_train, acc_test, loss_train, loss_test = train_baseline(64, 0.01, 0.9, 0.1, 0, 0.005, "sgd", "xavier", preform_whitening=False,depth=depth)
#     end = time.time()
#     acc_train_by_depth.append(acc_train)
#     loss_train_by_depth.append(loss_train)
#     acc_test_by_depth.append(acc_test)
#     loss_test_by_depth.append(loss_test)
#     print(f"runtime: {end - start} \n")
#
# print(f"loss_train_by_depth: {loss_train_by_depth} \nloss_test_by_depth: {loss_test_by_depth} \nacc_train_by_depth:"
#       f" {acc_train_by_depth} \nacc_test_by_depth: {acc_test_by_depth}")
#
# plot_res([3,4,10], acc_train_by_depth, 'train accuracy', "q2_acc_vs_depth", "depth", "Accuracy"
#          , "Accuracy vs Depth", Y2=acc_test_by_depth , label2='test accuracy')
#
# plot_res([3,4,10], loss_train_by_depth, 'train loss', "q2_loss_vs_depth", "depth", "Loss"
#          , "Loss vs Depth", Y2=loss_test_by_depth , label2='test loss')
import pickle
import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA

from utils import get_image_from_vector
from utils import *
import time


class Baseline(nn.Module):
    def __init__(self, width, sd, dropout, init_type, depth, filter_size1=64, filter_size2=16):
        super(Baseline, self).__init__()
        self.skip = nn.Identity()
        self.depth = depth
        self.conv1 = nn.Conv2d(3, filter_size1, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        if depth == 2:
            self.conv2 = nn.Conv2d(filter_size1, filter_size2, kernel_size=3, stride=1)
            self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        if depth == 3:
            self.conv2 = nn.Conv2d(filter_size1, filter_size1, kernel_size=3, stride=1)
            self.conv3 = nn.Conv2d(filter_size1, filter_size2, kernel_size=3, stride=1)
            self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        elif depth == 4:
            self.conv2 = nn.Conv2d(filter_size1, filter_size1, kernel_size=3, stride=1)
            self.conv3 = nn.Conv2d(filter_size1, filter_size1, kernel_size=3, stride=1)
            self.conv4 = nn.Conv2d(filter_size1, filter_size2, kernel_size=3, stride=1)
            self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        elif depth == 5:
            self.conv2 = nn.Conv2d(filter_size1, filter_size1, kernel_size=3, stride=1)
            self.conv3 = nn.Conv2d(filter_size1, filter_size1, kernel_size=3, stride=1)
            self.conv4 = nn.Conv2d(filter_size1, filter_size1, kernel_size=3, stride=1)
            self.conv5 = nn.Conv2d(filter_size1, filter_size2, kernel_size=3, stride=1)
            self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.linear_sizes = {
            2:filter_size2*7*7,
            3:filter_size2*3*3,
            4:filter_size2*2*2,
            5:filter_size2
        }
        self.linear = nn.Linear(self.linear_sizes[depth], 10)

        if init_type == "xavier":
            torch.nn.init.xavier_uniform_(self.linear.weight)
            torch.nn.init.xavier_uniform_(self.conv1.weight)
            torch.nn.init.xavier_uniform_(self.conv2.weight)
        elif init_type == "normal":
            self.linear.weight.data.normal_(0.0, sd)
            self.conv1.weight.data.normal_(0.0, sd)
            self.conv2.weight.data.normal_(0.0, sd)
            if depth > 2:
                self.conv3.weight.data.normal_(0.0, sd)
            if depth > 3:
                self.conv4.weight.data.normal_(0.0, sd)
            if depth > 4:
                self.conv5.weight.data.normal_(0.0, sd)

    def forward(self, x):
        depth = self.depth
        original_input = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        if self.skip:
            x = x + self.skip(original_input)
        x = self.relu(x)
        x = self.max_pool2(x)

        if depth == 3:
            x = self.conv3(x)
            x = self.relu(x)
            x = self.max_pool3(x)

        if depth == 4:
            x = self.conv3(x)
            x = self.conv4(x)
            if self.skip:
                x = x + self.skip(original_input)
            x = self.relu(x)
            x = self.max_pool3(x)

        if depth == 5:
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            if self.skip:
                x = x + self.skip(original_input)
            x = self.relu(x)
            x = self.max_pool3(x)

        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name, preform_whitening):
        self.images, self.labels = self.load_images(file_name, preform_whitening)

    def load_images(self, file_name, preform_whitening=False):
        with open(file_name, "rb") as df:
            loaded = pickle.load(df)
            data = loaded["data"]
            data = [get_image_from_vector(im) for im in data]
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


def train_baseline(width, lr, momentum, sd, wd, dropout, optimizer_type, init_type, preform_whitening, plot_name="plot",
                   filter_size1=64, filter_size2=16, depth=2):
    batch_size = 1024

    device = get_training_device()
    model = Baseline(width, sd, dropout, init_type, depth=depth, filter_size1=filter_size1,
                     filter_size2=filter_size2).to(device)

    train_set = Dataset("./data/train.pkl", preform_whitening=preform_whitening)
    test_set = Dataset("./data/test.pkl", preform_whitening=False)
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

    num_epoch = 500
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

    best_epoch = acc_test.index(max(acc_test))
    print("Best Epoch: {}/{}, Train Accuracy: {}%, Test accuracy: {}% , Train loss: {} , Test loss: {}".format(
        best_epoch,
        num_epoch,
        round(acc_train[best_epoch] * 100, 2),
        round(acc_test[best_epoch] * 100, 2),
        round(loss_train[best_epoch], 3),
        round(loss_test[best_epoch], 3)
    ))

    print("Last Epoch: {}, Train Accuracy: {}%, Test accuracy: {}% , Train loss: {} , Test loss: {}".format(
        len(acc_test) - 1,
        round(acc_train[-1] * 100, 2),
        round(acc_test[-1] * 100, 2),
        round(loss_train[-1], 3),
        round(loss_test[-1], 3)
    ))

    torch.save(model.state_dict(), "./models/q2_base.pt")
    plot_loss_vs_epoch(loss_train, plot_name + "_Q3_loss_vs_epoch", loss_test=loss_test)
    plot_acc_vs_epoch(acc_train, plot_name + "_Q3_acc_vs_epoch", acc_test=acc_test)

    return acc_train, acc_test, loss_train, loss_test


# # Q3 - baseline grid search
# best_accuracy = 0
# params = []
# for lr in [0.1, 0.01, 0.001]:
#     for m in [0.9, 0.5, 0.1, 0]:
#         for sd in [1.0, 0.5, 0.25, 0.1]:
#             print(f"running with lr:{lr}, m:{m}, sd:{sd}")
#             test_accuracy = train_baseline(None, lr, m, sd,0, None, "sgd", "normal", False,f"Baseline_lr:{lr}_m:{m}_sd:{sd}")
#             if test_accuracy > best_accuracy:
#                 best_accuracy = test_accuracy
#                 params = [lr, m, sd]
# print(f"best_accuracy: {best_accuracy}")
# print(f"lr: {params[0]}, momentum:{params[1]}, sd: {params[2]}")
# # best configuration: lr=0.01, momentum=0.9 sd=0.1

lr = 0.01
m = 0.9
sd = 0.1

# # compare optimizers:
# print(f"Adam: lr={lr}, momentum={m} sd={sd}")
# train_baseline(None, lr,m, sd,0, 0, "adam", "normal", False, "Adam_optimizer")
# print(f"SGD: lr={lr}, momentum={m} sd={sd}")
# train_baseline(None, lr, m, sd, 0, 0, "sgd", "normal", False, "SGD_optimizer")
# # best optimizer is SGD
#
#
# #compare init values:
# print(f"Xavier: SGD, lr={lr}, momentum={m} sd={sd}")
# train_baseline(None,  lr,m, sd,0,0, "sgd", "xavier", False, "Xavier_init")
# # Normal initialization gives better results
#
#
# compare regularization
# print(f"comparing regularization,using SGD, lr={lr}, momentum={m} sd={sd}\n")
# for dropout in [0, 0.001, 0.005, 0.009]:
#     for wd in [0, 0.001, 0.005, 0.009]:
#         print(f"dropout: {dropout}, weight decay: {wd}:")
#         start = time.time()
#         train_baseline(None, lr, m, sd, wd, dropout, "sgd", "normal",False, f"regularization_{dropout}_{wd}")
#         end = time.time()
#         print(f"runtime: {round(end - start)} \n")

#
# # preprocessing
# print(f"Preforming PCA whitening: SGD, lr={lr}, momentum={m} sd={sd}")
# start = time.time()
# train_baseline(None, lr, m, sd, 0, 0, "sgd", "normal", True, name="PCA_whitening")
# end = time.time()
# print(f"runtime: {round(end - start)} \n")
#
#
# compare width:
# acc_train_by_width = []
# loss_train_by_width = []
# acc_test_by_width = []
# loss_test_by_width = []
# for (size1, size2) in [(64, 16), (256, 64), (512, 256)]:
#     print(f"width ({size1},{size2}): SGD, lr={lr}, momentum={m} sd={sd}, dropout=0, wd=0")
#     acc_train, acc_test, loss_train, loss_test = train_baseline(None, lr, m, sd, 0, 0, "sgd", "normal", False, f"filter_size_({size1},{size2})", size1, size2)
#     acc_train_by_width.append(acc_train)
#     loss_train_by_width.append(loss_train)
#     acc_test_by_width.append(acc_test)
#     loss_test_by_width.append(loss_test)
#
# arrays = [acc_test_by_width, acc_train_by_width, loss_test_by_width, loss_train_by_width]
# with open('./width_arrays.txt', 'wb') as f:
#     pickle.dump(arrays, f)

# with open('./width_arrays.txt', 'rb') as f:
#     arrays = pickle.load(f)
# acc_test_by_width, acc_train_by_width, loss_test_by_width, loss_train_by_width = arrays
#
#
# plot_res(range(500), acc_train_by_width[0], 'train accuracy (64,16)', "Q3_acc_vs_width", "Width", "Accuracy"
#          , "Accuracy vs Width", Y2=acc_test_by_width[0], label2='test accuracy (64,16)', Y3=acc_train_by_width[1]
#          , label3='train accuracy (256,64)', Y4=acc_test_by_width[1], label4='test accuracy (256,64)', Y5=acc_train_by_width[2]
#          , label5='train accuracy (512,256)', Y6=acc_test_by_width[2], label6='test accuracy (512,256)',acc=True)
#
# plot_res(range(500), loss_train_by_width[0], 'train loss (64,16)', "Q3_loss_vs_width", "Width", "Loss"
#          , "Loss vs Width", Y2=loss_test_by_width[0], label2='test loss (64,16)', Y3=loss_train_by_width[1]
#          , label3='train loss (256,64)', Y4=loss_test_by_width[1], label4='test loss (256,64)', Y5=loss_train_by_width[2]
#          , label5='train loss (512,256)', Y6=loss_test_by_width[2], label6='test loss (512,256)',acc=False)


#
# compare depth:
# acc_train_by_depth = []
# loss_train_by_depth = []
# acc_test_by_depth = []
# loss_test_by_depth = []
# for k in [2,3,4,5]:
#     print(f"{k} convolution layers: SGD, lr={lr}, momentum={m} sd={sd}, dropout=0, wd=0")
#     acc_train, acc_test, loss_train, loss_test = train_baseline(None, lr, m, sd, 0, 0, "sgd", "normal", False, f"{k}_convolution_layers",depth=k)
#     acc_train_by_depth.append(acc_train)
#     loss_train_by_depth.append(loss_train)
#     acc_test_by_depth.append(acc_test)
#     loss_test_by_depth.append(loss_test)
#
# arrays = [acc_test_by_depth, acc_train_by_depth, loss_test_by_depth, loss_train_by_depth]
# with open('./depth_arrays.txt', 'wb') as f:
#     pickle.dump(arrays, f)


with open('./depth_arrays.txt', 'rb') as f:
    arrays = pickle.load(f)
acc_test_by_depth, acc_train_by_depth, loss_test_by_depth, loss_train_by_depth = arrays

plot_res(range(500), acc_train_by_depth[0], 'train accuracy k=2', "Q3_acc_vs_depth", "Depth", "Accuracy"
         , "Accuracy vs Depth", Y2=acc_test_by_depth[0], label2='test accuracy k=2', Y3=acc_train_by_depth[1]
         , label3='train accuracy k=3', Y4=acc_test_by_depth[1], label4='test accuracy k=3', Y5=acc_train_by_depth[2]
         , label5='train accuracy k=4', Y6=acc_test_by_depth[2], label6='test accuracy k=4', Y7=acc_train_by_depth[3]
         , label7='train accuracy k=5', Y8=acc_test_by_depth[3], label8='test accuracy k=5',acc=True)

plot_res(range(500), loss_train_by_depth[0], 'train loss k=2', "Q3_loss_vs_depth", "Depth", "Loss"
         , "Loss vs Depth", Y2=loss_test_by_depth[0], label2='test loss k=2', Y3=loss_train_by_depth[1]
         , label3='train loss k=3', Y4=loss_test_by_depth[1], label4='test loss k=3', Y5=loss_train_by_depth[2]
         , label5='train loss k=4', Y6=loss_test_by_depth[2], label6='test loss k=4', Y7=loss_train_by_depth[3]
         , label7='train loss k=5', Y8=loss_test_by_depth[3], label8='test loss k=5', acc=False)


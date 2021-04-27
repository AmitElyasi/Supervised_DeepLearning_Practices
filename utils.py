import pickle
import torch
import torch.nn as nn
import numpy as np

import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt


def get_image_from_vector(vec):
    rgb = [vec[:1024], vec[1024:2048], vec[2048:]]
    img_arr = []
    for color in rgb:
        color_arr = []
        for start_ind in range(0, 1024, 32):
            row = color[start_ind:start_ind + 32]
            color_arr.append(row)
        img_arr.append(color_arr)
    img_arr = np.array(img_arr)
    # img_arr = np.transpose(img_arr)
    return img_arr


def get_training_device(cuda_num=1):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_num}")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    return device


def load_dataset(file_name):
    with open(file_name, "rb") as f:
        loaded = pickle.load(f)
        data = loaded["data"]
        labels = loaded["labels"]
        return data, labels


def plot_loss_vs_epoch(loss_train, question, loss_test=None):
    plt.figure(0)
    plt.plot(loss_train, label='train loss')
    if loss_test:
        plt.plot(loss_test, label='test loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.savefig(f"./plots/{question}_loss_vs_epoch.png")
    plt.close()


def plot_acc_vs_epoch(acc_train, question, acc_test=None):
    plt.figure()
    plt.plot(acc_train, label='train accuracy')
    if acc_test:
        plt.plot(acc_test, label='test accuracy')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.savefig(f"./plots/{question}_acc_vs_epoch.png")
    plt.close()


def plot_res(X, Y, label, name, X_lab, Y_lab, title, Y2=None, label2=None, Y3=None
             , label3=None,Y4=None, label4=None,Y5=None, label5=None,Y6=None, label6=None
             ,Y7=None, label7=None,Y8=None, label8=None, acc=False):
    plt.figure()
    plt.plot(X,Y, '-', label=label)
    if Y2:
        plt.plot(X, Y2,'-', label=label2)
        plt.ylim([0,max(max(Y),max(Y2))+1])
    plt.xlim([X[0]-1,X[-1]+1])

    if Y3:
        plt.plot(X, Y3, '-', label=label3)
        plt.plot(X, Y4, '-', label=label4)
        plt.plot(X, Y5, '-', label=label5)
        plt.plot(X, Y6, '-', label=label6)

    if Y8:
        plt.plot(X, Y7, '-', label=label7)
        plt.plot(X, Y8, '-', label=label8)

    if acc:
        plt.ylim([0,1])
    else:
        plt.ylim([0, 5])

    plt.xlabel(X_lab)
    plt.ylabel(Y_lab)
    plt.title(title)
    plt.legend()
    plt.savefig(f"./plots/{name}.png")

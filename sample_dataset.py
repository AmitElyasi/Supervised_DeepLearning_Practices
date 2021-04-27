import pickle
import random
from PIL import Image
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def show_image_from_vector(vec):
    rgb = [vec[:1024], vec[1024:2048], vec[2048:]]
    img_arr = []
    for color in rgb:
        color_arr = []
        for start_ind in range(0, 1024, 32):
            row = color[start_ind:start_ind+32]
            color_arr.append(row)
        img_arr.append(color_arr)
    img_arr = np.array(img_arr)
    img_arr = np.transpose(img_arr)
    print(img_arr.shape)
    Image.fromarray(img_arr).show()


batch_file = './cifar-10-batches-py/data_batch_{}'
train_data_file = "./data/train.pkl"
test_data_file = "./data/test.pkl"

choose_for_batch = 1000
original_batch_size = 10000
all_data = []
all_labels = []


for i in range(1,6):
    file_name = batch_file.format(i)
    res = unpickle(file_name)
    data = res[b"data"]
    labels = res[b"labels"]
    indices = random.sample([i for i in range(original_batch_size)], choose_for_batch)

    for ind in indices:
        img = data[ind]/255
        all_data.append(img)
        all_labels.append(labels[ind])


all_data = np.array(all_data)
with open(train_data_file, "wb") as f:
    pickle.dump(dict(data=all_data, labels=all_labels), f)


test_data = []
test_labels = []

res = unpickle('./cifar-10-batches-py/test_batch')
data = res[b"data"]
labels = res[b"labels"]
indices = random.sample([i for i in range(original_batch_size)], choose_for_batch)

for ind in indices:
    img = data[ind]/255
    test_data.append(img)
    test_labels.append(labels[ind])

test_data = np.array(test_data)
with open(test_data_file, "wb") as f:
    pickle.dump(dict(data=test_data, labels=test_labels), f)


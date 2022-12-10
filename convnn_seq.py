# imports
from matplotlib import style
import matplotlib.pyplot as plt
import time
from math import sqrt
from tqdm import tqdm
from numpy.random import Generator, PCG64
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.cuda.amp import GradScaler
torch.cuda.is_available()


print(torch.__version__)
# this code trains the sequential model

# Generator
rg = Generator(PCG64())

# device for training
device = torch.device("cuda:0")
#device = torch.device("cpu")
# Network architecture
net = nn.Sequential(
    nn.Dropout(p=0.2),
    # input shape (Batch_Size, channles=1, H=32, W=32)
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3),
    # input shape (Batch_Size, channles=6, H=30, W=30)
    nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3),
    # input shape (Batch_Size, channles=6, H=28, W=28)
    nn.ReLU(),
    nn.BatchNorm2d(6),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # input shape (Batch_Size, channles=6, H=14, W=14)
    nn.Dropout(p=0.2),
    nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
    # input shape (Batch_Size, channles=12, H=10, W=10),
    nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5),
    # input shape (Batch_Size, channles=12, H=6, W=6),
    nn.ReLU(),
    nn.BatchNorm2d(12),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # input shape (Batch_Size, channles=12, H=3, W=3)
    nn.Flatten(start_dim=1),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=12 * 3 * 3, out_features=108),
    nn.ReLU(),
    nn.BatchNorm1d(108),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=108, out_features=54),
    nn.ReLU(),
    nn.BatchNorm1d(54),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=54, out_features=6),
)

net = net.to(device)

optimizer = optim.AdamW(net.parameters(), lr=0.001,
                        amsgrad=True, weight_decay=0.01)

loss_function = nn.MSELoss()

# load training data
training_data = np.load(
    "C:\\Users\gabri\OneDrive\Documents\Code/training_data_v2.npy", allow_pickle=True)

# shuffle training data
rg.shuffle(training_data)


image = torch.Tensor([i[0] for i in training_data]).view(-1, 32, 32)

# send image to device
image = image.to(device)

label = torch.Tensor([i[1] for i in training_data])

# send label to device
label = label.to(device)


VAL_PCT = 0.3  # lets reserve 30% of our data for validation
val_size = int(len(image) * VAL_PCT)

# List with all model names to build the statistics
model_names = []

BATCH_SIZE = 4
EPOCHS = 400
NUM_CV = 5

GRADIENT_ACCUM = 16

# Helper methods


def fwd_pass(network, X, y, train=False):
    outputs = network(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True) / len(matches)

    loss = loss_function(outputs, y)

    if train:
        scaler.scale(loss/GRADIENT_ACCUM).backward()
        optimizer.step()

    return acc, loss


scaler = GradScaler()

# Training loop


def train(network, model_name):
    with open("model_v14.log", "a") as f:
        for epoch in range(EPOCHS):
            print(f"This is epoch:{epoch}\n")
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i: i + BATCH_SIZE].view(-1, 1, 32, 32)
                batch_y = train_Y[i: i + BATCH_SIZE]

                if batch_X is None:
                    print("returned batch_X is None")
                if batch_y is None:
                    print("returned batch_y is None")

                #batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                network.zero_grad()
                acc, loss = fwd_pass(network, batch_X, batch_y, train=True)

                if acc is None:
                    print("returned acc is None")
                if loss is None:
                    print("returned loss is None")

                if (i + 1) % GRADIENT_ACCUM == 0:
                    scaler.step(optimizer)
                    scaler.update()

                    network.zero_grad()

            # saving performance params every epoch:
            val_acc, val_loss = fwd_pass(
                network, val_X.view(-1, 1, 32, 32), val_Y)
            f.write(
                f"{model_name},{epoch},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n"
            )
            print(
                f"{model_name},{epoch},train_acc: {round(float(acc),2)},val_acc: {round(float(val_acc),2)}\n")


# Main crossvalidation loop
for cv_index in range(NUM_CV):
    # Sampling the test and validation sets
    pic_mask = np.array(np.zeros(60000, dtype=bool))
    index_train = rg.choice(60000, size=round(
        60000 * (1 - VAL_PCT)), replace=False)
    pic_mask[index_train] = True
    train_X = image[pic_mask, :]
    train_Y = label[pic_mask]
    val_X = image[np.logical_not(pic_mask), :]
    val_Y = label[np.logical_not(pic_mask)]
    # Reseting the parameters of the network after each cv epoch
    for mod in net.modules():
        if isinstance(mod, torch.nn.Linear) or isinstance(mod, torch.nn.Conv2d):
            mod.reset_parameters()
    optimizer.zero_grad()
    model_name = (
        f"model-cv-{cv_index}"  # gives a dynamic model name.
    )
    model_names.append(model_name)
    print(f"This is cv:{cv_index}\n")
    train(net, model_name)


def create_acc_loss_graph(model_namess_list):
    contents = open("model_v14.log", "r").read().split("\n")
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []
    for model_name in model_namess_list:
        temp_accuracies = []
        temp_losses = []
        temp_val_accs = []
        temp_val_losses = []

        for c in contents:
            if model_name in c:
                name, epoch, acc, loss, val_acc, val_loss = c.split(",")

                temp_accuracies.append(float(acc))
                temp_losses.append(float(loss))

                temp_val_accs.append(float(val_acc))
                temp_val_losses.append(float(val_loss))
        accuracies.append(temp_accuracies)
        losses.append(temp_losses)
        val_accs.append(temp_val_accs)
        val_losses.append(temp_val_losses)

    accuracies = np.array(accuracies)
    losses = np.array(losses)
    val_accs = np.array(val_accs)
    val_losses = np.array(val_losses)
    x_axis = range(EPOCHS)

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    means = np.mean(accuracies, axis=0)
    stds = np.std(accuracies, axis=0)
    ax1.plot(x_axis, means, label="acc", color="tab:blue")
    ax1.fill_between(
        x_axis, means - stds, means + stds, alpha=0.3, color="tab:blue"
    )

    means = np.mean(val_accs, axis=0)
    stds = np.std(val_accs, axis=0)
    ax1.plot(x_axis, means, label="val_acc", color="tab:red")
    ax1.fill_between(
        x_axis, means - stds, means + stds, alpha=0.3, color="tab:red"
    )
    ax1.legend(loc=2)

    means = np.mean(losses, axis=0)
    stds = np.std(losses, axis=0)
    ax2.plot(x_axis, means, label="losses", color="tab:blue")
    ax2.fill_between(
        x_axis, means - stds, means + stds, alpha=0.3, color="tab:blue"
    )

    means = np.mean(val_losses, axis=0)
    stds = np.std(val_losses, axis=0)
    ax2.plot(x_axis, means, label="val_loss", color="tab:red")
    ax2.fill_between(
        x_axis, means - stds, means + stds, alpha=0.3, color="tab:red"
    )
    ax2.legend(loc=2)
    plt.show()


create_acc_loss_graph(model_names)

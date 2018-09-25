import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.data

import dataset
from misc import *
from model import AlexNet

# ===========================================================
# Metadata & hyper-parameter setting up
# ===========================================================
sys.path.insert(0, 'data/')
EPOCHS = 5
BATCH_SIZE = 8
GPU_IN_USE = torch.cuda.is_available()
self_device = torch.device('cuda' if GPU_IN_USE else 'cpu')


# ===========================================================
# arguments setting up
# ===========================================================
parser = argparse.ArgumentParser(description='Gwario experiment')
parser.add_argument('--epochs', type=int, default=EPOCHS)
parser.add_argument('--lr', type=int, default=0.01)  # learning rate
parser.add_argument('--bs', type=int, default=BATCH_SIZE)
args = parser.parse_args()


# ===========================================================
# model setting up
# ===========================================================
baseline_model = AlexNet()
transfer_model = AlexNet()
transfer_model.load_state_dict(torch.load('./pretrained.model'))

model = transfer_model
model.last_connect = nn.Linear(256, 5)
model = model.to(self_device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = func.binary_cross_entropy
cudnn.benchmark = GPU_IN_USE


def train(data_in, target_in):
    accuracy_list = list()
    loss_list = list()

    model.train()
    num_batches = data_in.shape[0] // args.bs

    for i in range(num_batches):
        data = data_in[args.bs * i:args.bs * (i + 1)].to(self_device)
        target = target_in[args.bs * i:args.bs * (i + 1)].to(self_device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        acc = accuracy(output.data, target.data)

        loss_list.append(loss.item())
        accuracy_list.append(acc)

        progress_bar(i, num_batches, 'Loss : {:.4f} | Acc: {:.4f}'.format(loss.item(), acc))

    return accuracy_list, loss_list


def test(data_in, target_in):
    accuracy_list = list()
    loss_list = list()

    model.eval()
    num_batches = data_in.shape[0] // args.bs

    with torch.no_grad():
        for i in range(num_batches):
            data = data_in[args.bs * i:args.bs * (i + 1)].to(self_device)
            target = target_in[args.bs * i:args.bs * (i + 1)].to(self_device)
            output = model(data)

            test_loss = criterion(output, target).item()
            acc = accuracy(output.data, target.data)
            loss_list.append(test_loss)
            accuracy_list.append(acc)

            progress_bar(i, num_batches, 'Loss : {:.4f} | Acc: {:.4f}'.format(test_loss, acc))

    return accuracy_list, loss_list


def accuracy(prediction, target):
    diff = np.sum(prediction.cpu().numpy() == target.cpu().numpy())
    diff /= len(target) * 5
    return diff


def save_model(epoch):
    if not os.path.exists('./models'):
        os.makedirs('./models')

    if (epoch + 1) % 100 == 0:
        torch.save(model, './models/epoch-{}.model'.format(epoch + 1))


def validate(epoch):
    data, target = dataset.retrieve_data()
    print("data array length : %d" % data.shape[0])
    print("log array length : %d" % target.shape[0])
    size_fold = data.shape[0] // 5

    to_torch = torch.from_numpy

    for i in range(epoch):
        print("\n===> epoch: %d/%d" % (i + 1, epoch))

        # prepare the data
        x_train_temp = to_torch(np.concatenate((data[:(i % 5) * size_fold], data[((i % 5) + 1) * size_fold:]), axis=0)).float()
        x_test_temp = to_torch(data[(i % 5) * size_fold: ((i % 5) + 1) * size_fold]).float()
        y_train_temp = to_torch(np.concatenate((target[:(i % 5) * size_fold], target[((i % 5) + 1) * size_fold:]), axis=0)).float()
        y_test_temp = to_torch(target[(i % 5) * size_fold: ((i % 5) + 1) * size_fold]).float()

        # train and test
        train_acc, train_loss = train(x_train_temp, y_train_temp)
        test_acc, test_loss = test(x_test_temp, y_test_temp)

        save_log(train_acc, train_loss, test_acc, test_loss)
        save_model(i)  # save the model on certain iterations


def save_log(train_acc, train_loss, test_acc, test_loss):
    train_info = {'Train Accuracy': train_acc, 'Train Loss': train_loss}
    test_info = {'Test Accuracy': test_acc, 'Test Loss': test_loss}
    record_info(train_info, './train.csv', mode='train')
    record_info(test_info, './test.csv', mode='test')


if __name__ == '__main__':
    validate(args.epochs)

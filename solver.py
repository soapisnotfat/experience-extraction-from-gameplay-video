from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.model import resnet152
from data import dataloader
from misc import record_info


class BackProp(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.GPU_IN_USE = torch.cuda.is_available()
        self.seed = config.seed
        self.train_loader = None
        self.test_loader = None

    def build_model(self):
        self.model = resnet152(pretrained=False)
        self.model.fc_action = nn.Linear(512 * 4, 10)

        self.criterion = nn.CrossEntropyLoss()

        if self.GPU_IN_USE:
            cudnn.benchmark = True
            self.model.cuda()
            self.criterion.cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

    def build_dataloader(self):
        self.train_loader, self.test_loader = dataloader.get_dataloader(self.batch_size, self.test_batch_size)

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        accuracy_list = list()

        progress = tqdm(self.train_loader)
        for batch_num, (data, target) in enumerate(progress):

            if self.GPU_IN_USE:
                data, target = Variable(data).cuda(), Variable(target).long().cuda()

            self.optimizer.zero_grad()
            prediction = self.model(data)
            target = torch.max(target, 1)[1]

            loss = self.criterion(prediction, target)
            loss.backward()

            self.optimizer.step()
            # self.scheduler.step(loss.data.cpu().numpy())

            train_loss += loss.data[0]
            train_correct += torch.max(prediction.data, 1)[1].eq(target.data).cpu().sum()
            total += data.size(0)

            accuracy_list.append(train_correct / total)

        print("    Average Loss: {:.4f} | Average Accuracy: {:.4f}".format(train_loss / len(self.train_loader),
                                                                           train_correct / total))
        return train_loss / len(self.train_loader), accuracy_list

    def test(self):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        accuracy_list = list()

        progress = tqdm(self.test_loader)
        for batch_num, (data, target) in enumerate(progress):

            if self.GPU_IN_USE:
                data, target = Variable(data).cuda(), Variable(target).long().cuda()

            self.optimizer.zero_grad()
            prediction = self.model(data)
            target = torch.max(target, 1)[1]

            loss = self.criterion(prediction, target)

            test_loss += loss.data[0]
            test_correct += torch.max(prediction.data, 1)[1].eq(target.data).cpu().sum()
            total += data.size(0)

            accuracy_list.append(test_correct / total)

        print("    Average Loss: {:.4f} | Average Accuracy: {:.4f}".format(test_loss / len(self.train_loader),
                                                                           test_correct / total))
        return test_loss / len(self.train_loader), accuracy_list

    def run(self):
        self.build_model()
        self.build_dataloader()
        for epoch in range(1, self.epochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))

            train_loss, train_accuracy_list = self.train()
            test_loss, test_accuracy_list = self.test()

            train_info = {'Train Accuracy': train_accuracy_list}
            test_info = {'Test Accuracy': test_accuracy_list}
            record_info(train_info, './backprop_train.csv', mode='train')
            record_info(test_info, './backprop_test.csv', mode='test')


class ImageNet(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.GPU_IN_USE = torch.cuda.is_available()
        self.seed = config.seed
        self.train_loader = None
        self.test_loader = None

    def build_model(self):
        self.model = resnet152(pretrained=True)
        self.model.fc_action = nn.Linear(512 * 4, 10)

        self.criterion = nn.CrossEntropyLoss()

        if self.GPU_IN_USE:
            cudnn.benchmark = True
            self.model.cuda()
            self.criterion.cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

    def build_dataloader(self):
        self.train_loader, self.test_loader = dataloader.get_dataloader(self.batch_size, self.test_batch_size)

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        accuracy_list = list()

        progress = tqdm(self.train_loader)
        for batch_num, (data, target) in enumerate(progress):

            if self.GPU_IN_USE:
                data, target = Variable(data).cuda(), Variable(target).long().cuda()

            self.optimizer.zero_grad()
            prediction = self.model(data)
            target = torch.max(target, 1)[1]

            loss = self.criterion(prediction, target)
            loss.backward()

            self.optimizer.step()
            # self.scheduler.step(loss.data.cpu().numpy())

            train_loss += loss.data[0]
            train_correct += torch.max(prediction.data, 1)[1].eq(target.data).cpu().sum()
            total += data.size(0)

            accuracy_list.append(train_correct / total)

        print("    Average Loss: {:.4f} | Average Accuracy: {:.4f}".format(train_loss / len(self.train_loader),
                                                                           train_correct / total))
        return train_loss / len(self.train_loader), accuracy_list

    def test(self):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        accuracy_list = list()

        progress = tqdm(self.test_loader)
        for batch_num, (data, target) in enumerate(progress):

            if self.GPU_IN_USE:
                data, target = Variable(data).cuda(), Variable(target).long().cuda()

            self.optimizer.zero_grad()
            prediction = self.model(data)
            target = torch.max(target, 1)[1]

            loss = self.criterion(prediction, target)

            test_loss += loss.data[0]
            test_correct += torch.max(prediction.data, 1)[1].eq(target.data).cpu().sum()
            total += data.size(0)

            accuracy_list.append(test_correct / total)

        print("    Average Loss: {:.4f} | Average Accuracy: {:.4f}".format(test_loss / len(self.train_loader),
                                                                           test_correct / total))
        return test_loss / len(self.train_loader), accuracy_list

    def run(self):
        self.build_model()
        self.build_dataloader()
        for epoch in range(1, self.epochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))

            train_loss, train_accuracy_list = self.train()
            test_loss, test_accuracy_list = self.test()

            train_info = {'Train Accuracy': train_accuracy_list}
            test_info = {'Test Accuracy': test_accuracy_list}
            record_info(train_info, './imagenet_train.csv', mode='train')
            record_info(test_info, './imagenet_test.csv', mode='test')


class TeacherStudent(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.GPU_IN_USE = torch.cuda.is_available()
        self.seed = config.seed
        self.train_loader = None
        self.test_loader = None

    def build_model(self):
        self.model = resnet152(pretrained=False)
        self.model.load_state_dict(torch.load('./models/model.tar'))
        self.model.fc_action = nn.Linear(512 * 4, 10)

        self.criterion = nn.CrossEntropyLoss()

        if self.GPU_IN_USE:
            cudnn.benchmark = True
            self.model.cuda()
            self.criterion.cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

    def build_dataloader(self):
        self.train_loader, self.test_loader = dataloader.get_dataloader(self.batch_size, self.test_batch_size)

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        accuracy_list = list()

        progress = tqdm(self.train_loader)
        for batch_num, (data, target) in enumerate(progress):

            if self.GPU_IN_USE:
                data, target = Variable(data).cuda(), Variable(target).long().cuda()

            self.optimizer.zero_grad()
            prediction = self.model(data)
            target = torch.max(target, 1)[1]

            loss = self.criterion(prediction, target)
            loss.backward()

            self.optimizer.step()
            # self.scheduler.step(loss.data.cpu().numpy())

            train_loss += loss.data[0]
            train_correct += torch.max(prediction.data, 1)[1].eq(target.data).cpu().sum()
            total += data.size(0)

            accuracy_list.append(train_correct / total)

        print("    Average Loss: {:.4f} | Average Accuracy: {:.4f}".format(train_loss / len(self.train_loader),
                                                                           train_correct / total))
        return train_loss / len(self.train_loader), accuracy_list

    def test(self):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        accuracy_list = list()

        progress = tqdm(self.test_loader)
        for batch_num, (data, target) in enumerate(progress):

            if self.GPU_IN_USE:
                data, target = Variable(data).cuda(), Variable(target).long().cuda()

            self.optimizer.zero_grad()
            prediction = self.model(data)
            target = torch.max(target, 1)[1]

            loss = self.criterion(prediction, target)

            test_loss += loss.data[0]
            test_correct += torch.max(prediction.data, 1)[1].eq(target.data).cpu().sum()
            total += data.size(0)

            accuracy_list.append(test_correct / total)

        print("    Average Loss: {:.4f} | Average Accuracy: {:.4f}".format(test_loss / len(self.train_loader),
                                                                           test_correct / total))
        return test_loss / len(self.train_loader), accuracy_list

    def run(self):
        self.build_model()
        self.build_dataloader()
        for epoch in range(1, self.epochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))

            train_loss, train_accuracy_list = self.train()
            test_loss, test_accuracy_list = self.test()

            train_info = {'Train Accuracy': train_accuracy_list}
            test_info = {'Test Accuracy': test_accuracy_list}
            record_info(train_info, './teacher_student_train.csv', mode='train')
            record_info(test_info, './teacher_student_test.csv', mode='test')

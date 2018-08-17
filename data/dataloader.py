import os
from os import listdir
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class TenTrainSet(Dataset):
    _all_videos = ('archery', 'breaststroke', 'crossbow', 'dance', 'dodge', 'fly', 'horse_riding', 'run', 'skydiving',
                   'waving_weapon')

    def __init__(self, image_dir, input_transform=None):
        super(TenTrainSet, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x)
                                for x in listdir(image_dir) if self.is_image_file(x) and self.is_valid(x)]

        self.input_transform = input_transform

    def __getitem__(self, index):
        file_name = self.image_filenames[index]
        input_image = load_img(file_name)

        target = file_name.split('/')[-1]
        target = target.split('_')
        if len(target) == 3:
            target = target[0]
        else:
            target = target[0] + '_' + target[1]

        if self.input_transform:
            input_image = self.input_transform(input_image)

        temp = np.zeros(len(self._all_videos))
        temp[self._all_videos.index(target)] = 1
        target = temp
        return input_image, target

    def __len__(self):
        return len(self.image_filenames)

    @staticmethod
    def is_image_file(file_name):
        return any(file_name.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

    @staticmethod
    def is_valid(file_name):
        file_index = int(file_name.split('_')[-1].split('.')[0])
        return file_index < 20 or file_index >= 37


class TenTestSet(Dataset):
    _all_videos = ('archery', 'breaststroke', 'crossbow', 'dance', 'dodge', 'fly', 'horse_riding', 'run', 'skydiving',
                   'waving_weapon')

    def __init__(self, image_dir, input_transform=None):
        super(TenTestSet, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x)
                                for x in listdir(image_dir) if self.is_image_file(x) and self.is_valid(x)]

        self.input_transform = input_transform

    def __getitem__(self, index):
        file_name = self.image_filenames[index]
        input_image = load_img(file_name)

        target = file_name.split('/')[-1]
        target = target.split('_')
        if len(target) == 3:
            target = target[0]
        else:
            target = target[0] + '_' + target[1]

        if self.input_transform:
            input_image = self.input_transform(input_image)

        temp = np.zeros(len(self._all_videos))
        temp[self._all_videos.index(target)] = 1
        target = temp
        return input_image, target

    def __len__(self):
        return len(self.image_filenames)

    @staticmethod
    def is_image_file(file_name):
        return any(file_name.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

    @staticmethod
    def is_valid(file_name):
        file_index = int(file_name.split('_')[-1].split('.')[0])
        return 20 <= file_index < 37


def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    return img


def get_dataloader(batch_size, test_batch_size, image_dir='./data/frames'):
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = TenTrainSet(image_dir=image_dir, input_transform=transform)
    test_set = TenTestSet(image_dir=image_dir, input_transform=transform)
    training_data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return training_data_loader, testing_data_loader

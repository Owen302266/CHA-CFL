import os
import torch
import glob
import tqdm
import numpy as np
import json
import random
from torchvision.transforms import CenterCrop

from utils import common

# file root
project_root = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)

common.set_random_seed(14641)  # random seed


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, name='soc', is_train=False, transform=None, shuffle=False):
        self.is_train = is_train
        self.name = name

        self.images = []
        self.low_labels = []
        self.mid_labels = []
        self.high_labels = []
        self.serial_number = []
        self.azimuth_angle = []
        self.shuffle_index = []

        self.transform = transform
        self._load_data(path)

    def __len__(self):
        return len(self.high_labels)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        _image = self.images[idx]
        high_label = self.high_labels[idx]
        mid_label = self.mid_labels[idx]
        low_label = self.low_labels[idx]
        _serial_number = self.serial_number[idx]
        _azimuth_angle = self.azimuth_angle[idx]

        if self.transform:
            _image = self.transform(_image)

        return _image, low_label, mid_label, high_label, _serial_number, _azimuth_angle

    def data_shuffler(self, image_list, label_list, group_num=1, random=None):
        """
        shuffle the input data
        """
        _data_label = []
        _data_image = []
        n = len(image_list)
        _sort = np.linspace(
            0,
            n,
            num=int(n / group_num),
            endpoint=False,
            retstep=False,
            dtype=None,
        )

        np.random.shuffle(_sort)
        for i in range(0, int(n / group_num)):
            for j in range(group_num):
                if int(_sort[i]) + j >= n:
                    break
                _data_image.append(image_list[int(_sort[i]) + j])
                _data_label.append(label_list[int(_sort[i]) + j])

        return _data_image, _data_label

    def list_generator(self, path, mode, extension, _random, single):
        """
        random shuffle the data list, if single is False, the random is inside the low_labels
        """

        extension_mode = '/*.npy' if extension == 'image' else '/*.json'
        list = []
        _list1 = []
        _list2 = []
        _list3 = []

        list1 = [0, 1, 6, 7, 9]
        list2 = [2, 3, 4, 8,]
        list3 = [5,]

        file_list = glob.glob(os.path.join(project_root, path, f'{self.name}/{mode}/*'))
        if single:
            for _, dir in enumerate(file_list):
                _list = glob.glob((dir + f'{extension_mode}'))
                common.set_random_seed(123)
                if _random:
                    random.shuffle(_list)
                list.extend(_list)
   
        else:
            for i, dir in enumerate(file_list):
                if i in list1:
                    _list1.extend(glob.glob((dir + f'{extension_mode}')))
                if i in list2:
                    _list2.extend(glob.glob((dir + f'{extension_mode}')))
                if i in list3:
                    _list3.extend(glob.glob((dir + f'{extension_mode}')))

            common.set_random_seed(123)
            if _random:
                random.shuffle(_list1)
                random.shuffle(_list2)
                random.shuffle(_list3)
            
            list.extend(_list1)
            list.extend(_list2)
            list.extend(_list3)

        return list

    def _load_data(self, path):
        """
        generate json and npy
        """

        mode = 'train' if self.is_train else 'test'

        image_list = self.list_generator(path, mode, 'image', True, True)
        label_list = self.list_generator(path, mode, 'label', True, True)

        # shuffle the data with groups
        if self.is_train:
            image_list, label_list = self.data_shuffler(
                image_list, label_list, group_num=3
            )

        for image_path, label_path in tqdm.tqdm(
            zip(image_list, label_list), desc=f'load {mode} data set'
        ):
            self.images.append(np.load(image_path))

            with open(label_path, mode='r', encoding='utf-8') as f:
                _label = json.load(f)

            self.low_labels.append(_label['low_class_id'])
            self.mid_labels.append(_label['mid_class_id'])
            self.high_labels.append(_label['high_class_id'])
            self.serial_number.append(_label['serial_number'])
            self.azimuth_angle.append(_label['azimuth_angle'])

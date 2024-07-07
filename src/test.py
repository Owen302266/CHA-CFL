import torch
import torchvision
from data import loader
from data import preprocess
import models
from utils import common
import numpy as np
import os

def load_model(save_name):
    """
    load the existing model
    """
    model_data = torch.load(save_name)
    print("model load success")
    return model_data


def load_dataset(path, is_train, name, batch_size):
    """
    inherit PyTorch Dataloader to load the dataset

    input:
        path: the path of the dataset
        is_train: whether the dataset is used for training, False for test loop 
        name: the name of the dataset, including soc, eoc, etc.
        batch_size: the batch size of the dataset, we use 128 for the train and test loop
    return:
        dataloader: dataloader inherit from PyTorch Dataloader
    """
    # preprocessing the dataset
    transform = [preprocess.CenterCrop(88), torchvision.transforms.ToTensor()]

    _dataset = loader.Dataset(
        path,
        name=name,
        is_train=is_train,
        transform=torchvision.transforms.Compose(transform),
    )
    # Dataloader
    data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )
    return data_loader


def validation(m, ds):
    """
    test loop, return accuracy of each hierarchy

    input: 
        m: test model
        ds: test dataset
    output:
       low_accuracy: accuracy of low hierarchy
       mid_accuracy: accuracy of mid hierarchy
       high_accuracy: accuracy of high hierarchy
    
    """

    num_data = 0
    low_corrects = 0
    mid_corrects = 0
    high_corrects = 0

    # model parameters get
    m.nets.eval()
    # use softmax to get the probability
    _softmax = torch.nn.Softmax(dim=1)
    # load the test dataset and get the predictions of each hierarchy
    for _, data in enumerate(ds):
        images, low_labels, mid_labels, high_labels, _, azimuth_angle = data
        images = torch.tensor(np.expand_dims(images[:, 0, :, :], axis=1))
        low_predictions, mid_predictions, high_predictions = m.inference(images)

        low_predictions = _softmax(low_predictions)
        mid_predictions = _softmax(mid_predictions)
        high_predictions = _softmax(high_predictions)
        # find the most possible one-hot label
        _, low_predictions = torch.max(low_predictions.data, 1)
        _, mid_predictions = torch.max(mid_predictions.data, 1)
        _, high_predictions = torch.max(high_predictions.data, 1)
        # transform the label to LongTensor
        low_labels = low_labels.type(torch.LongTensor)
        mid_labels = mid_labels.type(torch.LongTensor)
        high_labels = high_labels.type(torch.LongTensor)
        # calculate the correct samples
        num_data += low_labels.size(0)
        low_corrects += (low_predictions == low_labels.to(m.device)).sum().item()
        mid_corrects += (mid_predictions == mid_labels.to(m.device)).sum().item()
        high_corrects += (high_predictions == high_labels.to(m.device)).sum().item()

    low_accuracy = 100 * low_corrects / num_data
    mid_accuracy = 100 * mid_corrects / num_data
    high_accuracy = 100 * high_corrects / num_data

    return low_accuracy, mid_accuracy, high_accuracy


def run(
    dataset,
    classes,
    channels,
    batch_size,
    lr,
    lr_step,
    lr_decay,
    weight_decay,
    dropout_rate,
    model_name,
    train_view,): 

    """
    run the test loop
    """

    valid_set = load_dataset("dataset", False, dataset, batch_size)
    
    # model save path
    model_path = os.path.join(loader.project_root, 'pretrained_models')
    # run the model
    m = models.Model(
        classes=classes,
        dropout_rate=dropout_rate,
        channels=channels,
        lr=lr,
        lr_step=lr_step,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        train_view=train_view,
    )

    m.load(os.path.join(model_path, f"{model_name}.pth"))

    accuracy = validation(m, valid_set)
    return accuracy


if __name__ == '__main__':
    # parameters config
    sampling_rate = 10  # 10% - 50%
    dataset = 'soc'
    classes = 10
    channels = 1
    batch_size = 128

    lr = 1e-3
    lr_step = 500 
    lr_decay = 0.1

    weight_decay = 4e-3
    dropout_rate = 0.5
    train_view = 3

    model_name = f'SamplingRate_{sampling_rate}_dataset_{dataset}_view_{train_view}'

    accuracy = run(
        dataset,
        classes,
        channels,
        batch_size,
        lr,
        lr_step,
        lr_decay,
        weight_decay,
        dropout_rate,
        model_name,
        train_view,
    )

    print(accuracy)
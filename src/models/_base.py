import torch
import models.network
# from . import network
import sys
import os
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Model(object):
    def __init__(self, **params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 前向网络计算
        self.embedding = models.network.CNN_BiLSTM(
            classes=params.get('classes', 10),
            channels=params.get('channels', 1),
            dropout_rate=params.get('dropout_rate', 0.5),
            test_replication=params.get('test_replication', 1),
        )
        self.embedding.to(self.device)
        self.nets = nn.ModuleList(
            [
                models.network.ContrastNet(self.embedding),
                models.network.HierClassifyNet(self.embedding),
            ]
        )
        # 网络导入GPU
        self.nets.to(self.device)

    
    @torch.no_grad()
    def inference(self, x):
        low, mid, high, all = self.nets[1](self.embedding(x.to(self.device)))
        low = low + all[:, 0: 3]
        mid = mid + all[:, 3: 11]
        high = high + all[:, 11: 21]
        return low, mid, high


    def save(self, path):
        """
        model save
        """
        torch.save(self.nets.state_dict(), path)

    def load(self, path):
        """
        model load
        """
        self.nets.load_state_dict(torch.load(path))
        self.nets.eval()


def main():
    # pairs测试
    m = Model()
    images = torch.randn([128, 10])
    labels = torch.randint(0, 10, (128,))
    low, mid, high = m.inference(images)


if __name__ == "__main__":
    main()

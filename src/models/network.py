import torch.nn as nn
import torch
from torch.nn import functional as F
from . import _blocks
# if run network.py for test, use the import as below instead of the above one
# import _blocks


class CNN_BiLSTM(nn.Module):
    def __init__(self, **params):
        super(CNN_BiLSTM, self).__init__()
        self.dropout_rate = params.get("dropout_rate", 0.5)
        self.classes = params.get("classes", 10)
        self.channels = params.get("channels", 1)
        self.batch_size = params.get("batch_size", 2)
        self.test_replication = params.get("test_replication", 1)
        # weight initialization: kaiming method
        _w_init = params.get(
            "w_init", lambda x: nn.init.kaiming_normal_(x, nonlinearity="relu")
        )
        # bias  initialization: constant 0.1
        _b_init = params.get("b_init", lambda x: nn.init.constant_(x, 0.1))

        # Convolutional Neural Network for feature extraction
        self._layer = nn.Sequential(
            # layer1: input: 88 * 88 * 1, output: 42 * 42 * 16
            _blocks.Conv2DBlock(
                shape=[5, 5, self.channels, 16],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=_w_init,
                b_init=_b_init,
            ),
            # layer2: input: 42 * 42 * 16, output: 19 * 19 * 32
            _blocks.Conv2DBlock(
                shape=[5, 5, 16, 32],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=_w_init,
                b_init=_b_init,
            ),
            # layer3: input: 19 * 19 * 32, output: 7 * 7 * 64
            _blocks.Conv2DBlock(
                shape=[6, 6, 32, 64],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=_w_init,
                b_init=_b_init,
            ),
            # layer4: input: 7 * 7 * 64, output: 3 * 3 * 128
            _blocks.Conv2DBlock(
                shape=[5, 5, 64, 128],
                stride=1,
                padding="valid",
                activation="relu",
                w_init=_w_init,
                b_init=_b_init,
            ),
            # layer5: flatten the output
            nn.Flatten(start_dim=2, end_dim=-1),
        )

        # LSTM for sequence learning
        self.lstm = nn.LSTM(
            input_size=self.test_replication * 3 * 3,
            hidden_size=32,
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )

    def forward(self, x):
        x = self._layer(x)
        x = x.reshape(
            int(x.shape[0] * self.test_replication),
            int(x.shape[1] / self.test_replication),
            x.shape[2],
        )
        out, (_, _) = self.lstm(x)
        out = out.reshape(out.shape[0], self.test_replication * 128, 8, 8)
        return out


class ContrastNet(nn.Module):
    def __init__(self, embedding_net, **params):
        super(ContrastNet, self).__init__()
        self.dropout_rate = params.get("dropout_rate", 0.5)
        self.classes = params.get("classes", 10)
        self.channels = params.get("channels", 1)
        self.batch_size = params.get("batch_size", 2)
        self.test_replication = params.get("test_replication", 1)
        # weight initialization: kaiming method
        _w_init = params.get(
            "w_init", lambda x: nn.init.kaiming_normal_(x, nonlinearity="relu")
        )
        # bias initialization: constant 0.1
        _b_init = params.get("b_init", lambda x: nn.init.constant_(x, 0.1))

        self.embedding_net = embedding_net
        self.contrast_layer = _blocks.Conv2DBlock(
            shape=[8, 8, 128, 32],
            stride=1,
            padding="valid",
            w_init=_w_init,
            b_init=_b_init,
        )

    def forward(self, x):
        # x = self.embedding_net(x)
        x = self.contrast_layer(x)
        return x


class HierClassifyNet(nn.Module):
    def __init__(self, embedding_net, **params):
        super(HierClassifyNet, self).__init__()
        self.dropout_rate = params.get("dropout_rate", 0.5)
        self.low_classes = params.get("low_classes", 3)
        self.mid_classes = params.get("mid_classes", 8)
        self.high_classes = params.get("high_classes", 10)
        self.classes = self.low_classes + self.mid_classes + self.high_classes
        # weight初始化：何凯明方法
        _w_init = params.get(
            "w_init", lambda x: nn.init.kaiming_normal_(x, nonlinearity="relu")
        )
        # bias初始化：常数0.1
        _b_init = params.get("b_init", lambda x: nn.init.constant_(x, 0.1))

        self.embedding_net = embedding_net
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.low_layer = _blocks.Conv2DBlock(
            shape=[3, 3, 128, 64],
            stride=1,
            padding="same",
            w_init=_w_init,
            b_init=_b_init,
        )  # output: 8 * 8 * 128

        self.low_classifier = _blocks.Conv2DBlock(
            shape=[8, 8, 64, self.low_classes],
            stride=1,
            padding="valid",
            w_init=_w_init,
            b_init=_b_init,
        )

        self.mid_layer = _blocks.Conv2DBlock(
            shape=[3, 3, 192, 64],
            stride=1,
            padding="same",
            w_init=_w_init,
            b_init=_b_init,
        )  # output: 8 * 8 * 128

        self.mid_classifier = _blocks.Conv2DBlock(
            shape=[8, 8, 64, self.mid_classes],
            stride=1,
            padding="valid",
            w_init=_w_init,
            b_init=_b_init,
        )

        self.high_layer = _blocks.Conv2DBlock(
            shape=[3, 3, 192, 64],
            stride=1,
            padding="same",
            w_init=_w_init,
            b_init=_b_init,
        )

        self.high_classifier = _blocks.Conv2DBlock(
            shape=[8, 8, 64, self.high_classes],
            stride=1,
            padding="valid",
            w_init=_w_init,
            b_init=_b_init,
        )
        self.global_output = _blocks.Conv2DBlock(
            shape=[8, 8, 64, self.classes],
            stride=1,
            padding="valid",
            w_init=_w_init,
            b_init=_b_init,
        )

    def forward(self, x):
        x = self.dropout(x)
        x1 = self.low_layer(x)
        x1 = self.dropout(x1)
        output1 = self.low_classifier(x1)
        output1 = output1.reshape(output1.shape[0], self.low_classes)
        x2 = torch.cat((x, x1), dim=1)
        x2 = self.mid_layer(x2)
        x2 = self.dropout(x2)
        output2 = self.mid_classifier(x2)
        output2 = output2.reshape(output2.shape[0], self.mid_classes)
        x3 = torch.cat((x, x2), dim=1)
        x3 = self.high_layer(x3)
        x3 = self.dropout(x3)
        output3 = self.high_classifier(x3)
        output3 = output3.reshape(output3.shape[0], self.high_classes)
        global_output = self.global_output(x3)
        global_output = global_output.reshape(global_output.shape[0], self.classes)

        return output1, output2, output3, global_output


def main():
    # CNN_LSTM网络测试
    a = torch.rand([4, 1, 88, 88])
    net = CNN_BiLSTM()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    outputs = net.forward(a.to(device))
    print(outputs.shape)


if __name__ == "__main__":
    main()

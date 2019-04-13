import torch
from torch import nn


class LstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, result_classes_count, num_layer=1):
        super(LstmNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, result_classes_count)

    def forward(self, x, hidden=None):
        '''input = (batch_size, sequence_size, features_count)'''
        lstm_output, (h_n, c_n) = self.lstm(x, hidden)
        result = self.out(lstm_output[:, -1, :])
        return result


if __name__ == "__main__":
    batch_size = 32
    input_size = 5000
    sequence_size = 4
    hidden_size = 25

    net = LstmNet(input_size=input_size, hidden_size=hidden_size, result_classes_count=100)
    x = torch.rand(batch_size, sequence_size, input_size)
    hidden = (torch.rand(1, batch_size, hidden_size), torch.rand(1, batch_size, hidden_size))
    output = net(x, hidden)
    print(output.size())

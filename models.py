import torch
import sys
import pandas as pd
from time import time
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPS = 1e-5
logger.remove()
logger.add(sys.stderr, level='DEBUG')


class DensityEstimator(nn.Module):
    def __init__(self, input_dim):
        """
        input of LSTM (seq_len, batch, input_size)
        initial hidden_state h0 or c0 = (num_layers * num_directions, batch, hidden_size)
        output of LSTM:  output, (h_n, c_n)
        output = (seq_len, batch, num_directions * hidden_size), here num_directions = 1
        """
        super(DensityEstimator, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = 128
        self.num_layers = 2

        self.representation = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim)
        )

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x, prev_state=None):
        """

        :param x: of shape (batch, input_size)
        :param prev_state:
        :return:
        """
        if prev_state is None:
            x = self.representation(x)
            output, state = self.lstm(x.view(1, -1, self.input_dim))
        else:
            output, state = self.lstm(x.view(1, -1, self.input_dim), prev_state)
        logits = self.fc(output.view(-1, self.hidden_size))
        return logits.softmax(dim=1), state

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

    def train_model(self, data_loader: DataLoader):
        """
        use cross entropy  loss
        """
        self.train()
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        start_time = time()
        for epoch in range(100):
            logger.debug(f'Epoch {epoch} =======================')
            optimizer.zero_grad()
            prev_state = None
            pred = None
            loss = torch.tensor(0.)
            for (x, y) in data_loader:
                missing = x.le(0)
                if pred is not None and missing.any():
                    x[missing] = pred.detach()[missing]  # fill missing inputs
                x, y = x.to(device), y.to(device)
                pred, prev_state = self(x, prev_state)
                # obs = torch.masked_select(- y * (pred + EPS).log(), y.ge(0))  # select observed data
                obs = torch.masked_select( (pred-y)**2, y.ge(0))  # select observed data
                loss += obs.sum()
            logger.debug(f'Loss: {loss.detach():.3f}, used {time() - start_time:.3f}s')
            loss.backward()
            optimizer.step()

    def forecast(self, start, steps_ahead: int):
        self.eval()
        with torch.no_grad():
            res = []
            pre_state = None
            pred = start
            for i in range(steps_ahead):
                pred, pre_state = self(pred, pre_state)
                res.append(pred.view(-1))
            return res

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    seq = [torch.rand(10).softmax(dim=0) for i in range(8)]

    dataset = TensorDataset(torch.stack(seq[:-1]), torch.stack(seq[1:]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = DensityEstimator(10)
    model.train_model(dataloader)

    start = seq[0]

    model.forecast(start, 20)






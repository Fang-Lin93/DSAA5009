
import sys, os
import torch
from time import time
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader
from data_processing import FamilyData, NonElderlyData
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPS = 1e-5
logger.remove()
logger.add(sys.stderr, level='DEBUG')

default_config = {
    'tag': 'fam',
    'input_dim': 4,
    'alpha': 1e-3,
    'beta': 1e-5
}


class ForecastModel(nn.Module):
    def __init__(self, pre_train=False, **config):
        """
        input_dim gives # of the bins of the PL
        county_dim gives the number of counties

        """
        super(ForecastModel, self).__init__()
        if config is None:
            config = default_config
        self.config = config
        self.tag = config['tag']
        self.input_dim = config['input_dim']
        self.county_dim = 159
        self.hidden_size = 128
        self.num_layers = 5
        self.alpha = config['alpha']
        self.beta = config['beta']

        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 159 * self.input_dim)
        )

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        self.init_c = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_layers * self.hidden_size)
        )

        self.init_h = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_layers * self.hidden_size)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim)
        )

        if pre_train:
            try:
                self.load_model()
            except Exception as exp:
                logger.error(exp)

    def forward(self, x, prev_state=None, step=0):
        """
        x: of shape (batch, input_size)
        give
        """
        if prev_state is None:
            prev_state = self.init_state(x)

        output, state = self.lstm(x.view(1, -1, self.input_dim), prev_state)

        logits = self.fc(output.view(-1, self.hidden_size))

        return logits.softmax(dim=1), \
            self.decoder(torch.cat([logits, torch.tensor([[1-torch.exp(-torch.tensor(step) / 10)]])], dim=1)), state

    def init_state(self, initial_input):
        return self.init_c(initial_input).view(self.num_layers, -1, self.hidden_size).to(device), \
               self.init_h(initial_input).view(self.num_layers, -1, self.hidden_size).to(device)

    def train_model(self, data_loader: DataLoader):
        """
        use cross entropy  loss
        """
        self.train()
        lr = 1e-3
        optimizer = optim.Adam(self.parameters(), lr=lr)
        start_time = time()
        for epoch in range(500):
            if lr % 100 == 0:
                lr *= 0.8
                optimizer = optim.Adam(self.parameters(), lr=lr)
            optimizer.zero_grad()
            prev_state = None
            loss = torch.tensor(0.)
            for i, (x, y, t, c) in enumerate(data_loader):
                # missing = x.le(0)
                # if pred is not None and missing.any():
                #     x[missing] = pred.detach()[missing]  # fill missing inputs
                x, y, t, c = x.to(device), y.to(device), t.to(device), c.to(device)
                pred_dens, pred_county, prev_state = self(x, prev_state, i)
                c_obs = torch.masked_select((pred_county - c) ** 2, c.ge(0))
                loss += (- y * pred_dens.log()).sum() + \
                        self.alpha * c_obs.sum() + self.beta * (pred_county.sum() - t.item()) ** 2
                # loss += (- y * pred_dens.log()).sum() + self.alpha * c_obs.sum() + \
                #     self.beta * (pred_county.sum() - t.item()) ** 2
            if epoch % 100 == 0:
                logger.debug(f'Epoch {epoch} loss: {loss.detach():.3f}, used {time() - start_time:.3f}s')
            loss.backward()
            optimizer.step()

    def forecast(self, start, steps_ahead: int):
        self.eval()
        with torch.no_grad():
            res_dens, res_county, res_total = [], [], []
            pre_state = self.init_state(start)
            pred_dens = start
            for i in range(steps_ahead):
                pred_dens, pred_county, pre_state = self(pred_dens, pre_state, i)
                res_dens.append(pred_dens.view(-1))
                res_county.append(pred_county.view(-1).relu())  # to remove negative values
                res_total.append((pred_county.sum()).int().item())
            return res_dens, res_county, res_total

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self):
        if not os.path.exists('models_dict'):
            os.mkdir('models_dict')
        torch.save(self.state_dict(), f'models_dict/{self.tag}_model')

    def load_model(self):
        self.load_state_dict(
            torch.load(f'models_dict/{self.tag}_model', map_location=torch.device('cpu')))


# if __name__ == '__main__':
#
#     F_model = ForecastModel(4)
#     F_model.train_model(FamilyData)
#
#     # NP_model = DensityEstimator(12)
#     # NP_model.train_model(NonElderlyData)
#
#     start = FamilyData.dataset[0][0]
#
#     # truth_den = [i[1] for i in FamilyData]
#     # truth_total = [i[2] for i in FamilyData]
#     # truth_county = [i[3] for i in FamilyData]
#
#     pred_dens, pred_county, pred_total = F_model.forecast(start, 20)
#
#     # truth_county[0][truth_county[0].ge(0)]
#     # pred_county[0][truth_county[0].ge(0)]

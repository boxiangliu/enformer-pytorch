from data.dataset import BasenjiDataset
from model.enformer import Enformer
from torch import nn, optim

class Trainer(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def initialize(self):
        self.iter = iter(self.data)
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="none")
        self.optimizer = optim.Adam(self.model.parameters())

    def train_step(self):
        try:
            batch = next(self.iter)

        except StopIteration:
            self.iter = iter(self.data)
            batch = next(self.iter)

        for head in ["human", "mouse"]:
            pred = self.model(batch[head]["sequence"])
            loss = self.criterion(pred, batch[head]["target"])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



data = BasenjiDataset()
model = Enformer()

trainer = Trainer(model, data)
trainer.train_step()
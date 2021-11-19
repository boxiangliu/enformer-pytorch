from data.dataset import BasenjiDataset
from model.enformer import Enformer
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer(object):

    def __init__(self, model, data, device="cuda"):
        self.model = model
        self.data = DataLoader(data, batch_size=1)
        self.device = device
        self.initialize()

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

            self.optimizer.zero_grad()

            pred = self.model(batch[head]["sequence"].to(self.device))
            loss = self.criterion(pred[head], batch[head]["target"].to(self.device))
            loss.mean().backward()
            print(f"head: {head}; loss: {loss.mean().item():03f}")
            self.optimizer.step()


def main():
    device = "cuda"
    data = BasenjiDataset(human_file="data/example_data_human.pt",
                          mouse_file="data/example_data_mouse.pt")
    model = Enformer().to(device)

    trainer = Trainer(model, data, device)

    for _ in range(20):
        trainer.train_step()

if __name__ == "__main__":
    main()
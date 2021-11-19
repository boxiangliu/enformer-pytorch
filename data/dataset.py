import os
import torch


class BasenjiDataset(torch.utils.data.Dataset):

    def __init__(self, human_file, mouse_file):
        self._human_file = human_file
        self._mouse_file = mouse_file
        self._human_data = torch.load(self._human_file)
        self._mouse_data = torch.load(self._mouse_file)

    @property
    def human_data(self):
        return self._human_data

    @property
    def mouse_data(self):
        return self._mouse_data

    def __len__(self):
        return len(self.human_data)

    def __getitem__(self, idx):
        return {"human": {"sequence": self.human_data["sequence"][idx],
                          "target": self.human_data["target"][idx]},
                "mouse": {"sequence": self.mouse_data["sequence"][idx],
                          "target": self.mouse_data["target"][idx]}}

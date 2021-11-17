import random 
import unittest
from model import enformer
import numpy as np
import torch

class TestEnformer(unittest.TestCase):

    def test_enformer(self):
        model = enformer.Enformer(channels=1536, num_heads=8, num_transformer_layers=11)
        inputs = _get_random_input()
        outputs = model(inputs)
        self.assertEqual(outputs["human"].shape, torch.Size((1, enformer.TARGET_LENGTH, 5313)))
        self.assertEqual(outputs["mouse"].shape, torch.Size((1, enformer.TARGET_LENGTH, 1643)))

def _get_random_input():
    seq = "".join(
        [random.choice("ACGT") for _ in range(enformer.SEQUENCE_LENGTH)])
    return np.expand_dims(enformer.one_hot_encode(seq), 0).astype(np.float32)

if __name__ == "__main__":
    unittest.main()
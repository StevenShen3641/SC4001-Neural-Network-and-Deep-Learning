import pickle
import numpy as np
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 2, 3])

print(torch.eq(a, b).type(torch.float).sum().item())


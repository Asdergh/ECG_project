import torch as th
import numpy as np
from torch import flatten
from torch.nn import Flatten



# flat = Flatten(end_dim=-2)
# test = th.normal(0.12, 1.12, (10, 32, 32, 32, 3))
# print(flatten(test, start_dim=1, end_dim=-2).size())

print(th.mul(th.tensor([1, 2, 3])))
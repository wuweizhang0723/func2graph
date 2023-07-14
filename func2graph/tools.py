import numpy as np
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F



# Implement two ways to look for weight matrix after training is done
# 1) take average of all N attention outputs as weight matrix
# 2) uses sliding windows so that we can visualize if the attention output is smoothly and continuously changed

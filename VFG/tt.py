"""
@Author: Shana
@File: tt.py
@Time: 11/20/20 4:35 PM
"""

import os
import torch
from torch_geometric.data import Data, Dataset

DIR = "./data/"

def get_geometric_data():
    x = torch.load(os.path.join(DIR, 'X'))
    y = torch.load(os.path.join(DIR, 'Y'))
    edge_index = torch.load(os.path.join(DIR, 'INDEX'))
    edge_attr = torch.load(os.path.join(DIR, 'ATTR'))

    data = Data(x=x,y=y,edge_index = edge_index.contiguous(),edge_attr =edge_attr)
    return data
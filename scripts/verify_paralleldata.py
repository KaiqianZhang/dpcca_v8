"""=============================================================================
Test PyTorch's ParallelData module.
============================================================================="""

import torch
import torch.nn as nn
from   torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return torch.device('cpu')

# ------------------------------------------------------------------------------

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# ------------------------------------------------------------------------------

class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        Lambda = torch.Tensor(output_size, input_size)
        self.Lambda = nn.Parameter(Lambda)

    def forward(self, x):
        y = torch.mm(self.Lambda, x.t())
        y = y + torch.ones(x.shape[0]).to(get_device())
        print('\tIn Model: input size', x.size(),
              'output size', y.size())
        return y

# ------------------------------------------------------------------------------

input_size  = 5
output_size = 2

batch_size  = 30
data_size   = 100

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

device = get_device()
model.to(device)

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

for data in rand_loader:
    x = data.to(device)
    y = model(x)
    print('Outside: input size', x.size(),
          'output_size', y.size())

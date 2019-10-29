"""=============================================================================
Script to verify that iterating over a dataset using PyTorch's Dataset,
DataLoader, and SubsetRandomSampler works as expected.
============================================================================="""

import math
import random

from   torch.utils.data.sampler import SubsetRandomSampler
from   torch.utils.data import DataLoader
from   torch.utils.data import Dataset

# ------------------------------------------------------------------------------

N_SAMPLES  = 100
CV_PCT     = 0.2
BATCH_SIZE = 4

# ------------------------------------------------------------------------------

class TestData(Dataset):

    def __init__(self):
        self.x = list(range(N_SAMPLES))

    def __getitem__(self, i):
        return self.x[i]

    def __len__(self):
        return len(self.x)

# ------------------------------------------------------------------------------

dataset = TestData()
indices = list(range(len(dataset)))
random.shuffle(indices)  # Shuffles in-place.

split = math.floor(N_SAMPLES * (1 - CV_PCT))
train_indices = indices[:split]
cv_indices = indices[split:]

# ------------------------------------------------------------------------------

train_loader = DataLoader(
    dataset,
    sampler=SubsetRandomSampler(train_indices),
    batch_size=BATCH_SIZE,
    drop_last=False
)

cv_loader = DataLoader(
    dataset,
    sampler=SubsetRandomSampler(cv_indices),
    batch_size=BATCH_SIZE,
    drop_last=False
)

# ------------------------------------------------------------------------------

train_samples_in_epoch = []
for x in train_loader:
    x1, x2, x3, x4 = x
    train_samples_in_epoch += [x1, x2, x3, x4]

cv_samples_in_epoch = []
for x in cv_loader:
    x1, x2, x3, x4 = x
    cv_samples_in_epoch += [x1, x2, x3, x4]

# print('=' * 80)
# train_samples_in_epoch.sort()
# print(len(train_samples_in_epoch))
# print(train_samples_in_epoch)
# print('=' * 80)
# print(len(cv_samples_in_epoch))
# print(cv_samples_in_epoch)
# print('=' * 80)

for x in cv_samples_in_epoch:
    assert x not in train_samples_in_epoch

samples_in_epoch = train_samples_in_epoch + cv_samples_in_epoch
samples_in_epoch.sort()
assert samples_in_epoch == dataset.x

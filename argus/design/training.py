from sklearn.linear_models import LogisticRegression, LinearRegression

from argus import ArgusDataset
from .utils import BaseTrainable

class LRTrainable(BaseTrainable):

    def step(self):

def train_lr(train: ArgusDataset, valid: ArgusDataset, test: ArgusDataset,)
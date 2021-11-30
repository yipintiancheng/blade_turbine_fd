from torch.utils.data import Dataset
from utils.dataset.datasetAugFun import *


class dataset(Dataset):

    def __init__(self, list_data, transform=None):
        self.seq_data = list_data[0]
        self.labels = list_data[1]
        if transform is None:
            self.transforms = Compose([
                nothing()
            ])
        else:
            self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        label = self.labels[item]
        seq = self.transforms(seq)
        return seq, label.astype(np.int64)

    def get_labels(self):
        return self.labels


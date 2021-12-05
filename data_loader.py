from torch.utils.data import Dataset
import pickle
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from arg_setting import args
from pathlib import Path


class IEMOCAPDataset(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.tensor(self.videoText[vid], dtype=torch.float32), \
               torch.tensor(self.videoVisual[vid], dtype=torch.float32), \
               torch.tensor(self.videoAudio[vid], dtype=torch.float32), \
               torch.tensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]], dtype=torch.float32), \
               torch.tensor([1] * len(self.videoLabels[vid]), dtype=torch.float32), \
               torch.tensor(self.videoLabels[vid], dtype=torch.int32), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in
                dat]


if __name__ == "__main__":
    data_path = Path(args.data_root) / (args.dataset + "_features_raw.pkl")
    dataset_test = IEMOCAPDataset(path=data_path)
    print(dataset_test[0])

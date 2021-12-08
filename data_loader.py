from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from arg_setting import args
from pathlib import Path
import pickle


def generate_utterance(path, data_set_train, data_set_test):
    text_, visual_, audio_, label_, vid_, idx_ = [], [], [], [], [], []
    for i in range(len(data_set_train)):
        text, visual, audio, _, _, label, vid = data_set_train[i]
        for idx in range(len(label)):
            text_.append(text[idx])
            visual_.append(visual[idx])
            audio_.append(audio[idx])
            label_.append(label[idx])
            vid_.append(vid)
            idx_.append(idx)

        # break

    data_utterance_train = {"text": text_,
                            "visual": visual_,
                            "audio": audio_,
                            "label": label_,
                            "vid": vid_,
                            "idx": idx_}

    text_, visual_, audio_, label_, vid_, idx_ = [], [], [], [], [], []
    for i in range(len(data_set_test)):
        text, visual, audio, _, _, label, vid = data_set_test[i]
        for idx in range(len(label)):
            text_.append(text[idx])
            visual_.append(visual[idx])
            audio_.append(audio[idx])
            label_.append(label[idx])
            vid_.append(vid)
            idx_.append(idx)

        # break

    data_set_test = {"text": text_,
                     "visual": visual_,
                     "audio": audio_,
                     "label": label_,
                     "vid": vid_,
                     "idx": idx_}
    info = {"train": data_utterance_train, "test": data_set_test}

    with open(path, 'wb') as f:
        pickle.dump(info, f)


class IEMOCAPDataset(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        lens = []
        for vid in self.keys:
            lens.append(len(self.videoLabels[vid]))
        self.lens = lens
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.tensor(self.videoText[vid], dtype=torch.float32), \
               torch.tensor(self.videoVisual[vid], dtype=torch.float32), \
               torch.tensor(self.videoAudio[vid], dtype=torch.float32), \
               torch.tensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]], dtype=torch.float32), \
               torch.tensor([1] * len(self.videoLabels[vid]), dtype=torch.float32), \
               torch.tensor(self.videoLabels[vid], dtype=torch.long), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in
                dat]


class IEMOCAPDatasetUtter:
    def __init__(self, path, device, train=True):
        cache = pickle.load(open(path, 'rb'), encoding='latin1')
        if train:
            cache = cache["train"]
        else:
            cache = cache["test"]

        self.text_ = cache["text"]
        self.visual_ = cache["visual"]
        self.audio_ = cache["audio"]
        self.label_ = cache["label"]
        self.vid_ = cache["vid"]
        self.idx_ = cache["idx"]
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.len = len(self.text_)
        self.device = device

    def get_data(self):
        text_view = torch.stack(self.text_, dim=0).to(self.device)
        visual_view = torch.stack(self.visual_, dim=0).to(self.device)
        audio_view = torch.stack(self.audio_, dim=0).to(self.device)
        return {"0": text_view, "1": visual_view, "2": audio_view}

    def get_label(self):
        labels = torch.tensor(self.label_, dtype=torch.long).unsqueeze(1).to(self.device)
        return labels

    def get_info(self):
        return self.vid_, self.idx_

    def __len__(self):
        return self.len

    # def __getitem__(self, index):
    #     return self.text_[index], \
    #            self.visual_[index], \
    #            self.audio_[index], \
    #            self.label_[index], \
    #            self.vid_[index], \
    #            self.idx_[index]


def get_loaders(train_set, test_set, batch_size, num_workers, pin_memory=False):
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              collate_fn=train_set.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=False)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             collate_fn=test_set.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    dataset_train = IEMOCAPDataset(path=args.data_path)
    dataset_test = IEMOCAPDataset(path=args.data_path, train=False)
    # d_T = 100, d_V=512, d_A = 100
    generate_utterance(args.utterance_path,
                       dataset_train,
                       dataset_test)
    dataset_utter_train = IEMOCAPDatasetUtter(args.utterance_path)
    dataset_utter_test = IEMOCAPDatasetUtter(args.utterance_path, train=False)
    print(len(dataset_utter_train))
    print(len(dataset_utter_test))
    print(dataset_train.keys)

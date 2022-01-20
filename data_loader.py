from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from arg_setting import args
from pathlib import Path
import pickle


def generate_utterance(path, data_set_train, data_set_test):
    r1_, r2_, r3_, r4_, label_, vid_, idx_ = [], [], [], [], [], [], []
    for i in range(len(data_set_train)):
        r1, r2, r3, r4, _, _, label, vid = data_set_train[i]
        for idx in range(len(label)):
            r1_.append(r1[idx])
            r2_.append(r2[idx])
            r3_.append(r3[idx])
            r4_.append(r4[idx])
            # audio_.append(audio[idx])
            label_.append(label[idx])
            vid_.append(vid)
            idx_.append(idx)

    data_utterance_train = {"r1": r1_,
                            "r2": r2_,
                            "r3": r3_,
                            "r4": r4_,
                            "label": label_,
                            "vid": vid_,
                            "idx": idx_}

    r1_, r2_, r3_, r4_, label_, vid_, idx_ = [], [], [], [], [], [], []
    for i in range(len(data_set_test)):
        r1, r2, r3, r4, _, _, label, vid = data_set_test[i]
        for idx in range(len(label)):
            r1_.append(r1[idx])
            r2_.append(r2[idx])
            r3_.append(r3[idx])
            r4_.append(r4[idx])
            # audio_.append(audio[idx])
            label_.append(label[idx])
            vid_.append(vid)
            idx_.append(idx)

    data_utterance_test = {"r1": r1_,
                           "r2": r2_,
                           "r3": r3_,
                           "r4": r4_,
                           "label": label_,
                           "vid": vid_,
                           "idx": idx_}

    info = {"train": data_utterance_train, "test": data_utterance_test}

    with open(path, 'wb') as f:
        pickle.dump(info, f)


class EmoryNlpDataset(Dataset):
    def __init__(self, path, dims, num_view, train=True):
        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainIds if train else self.testIds)]

        self.len = len(self.keys)
        self.lens = [len(self.labels[x]) for x in self.keys]

        self.Sn = None
        self.num_view = num_view

        self.keys_lens = {}
        self.h = {}
        for k, l in zip(self.keys, self.lens):
            self.keys_lens[k] = l
            self.h[k] = torch.zeros((l, dims[-1]), dtype=torch.float)

    def __getitem__(self, index):
        vid = self.keys[index]

        if self.Sn is None:
            r1 = torch.tensor(self.roberta1[vid], dtype=torch.float32)
            r2 = torch.tensor(self.roberta2[vid], dtype=torch.float32)
            r3 = torch.tensor(self.roberta3[vid], dtype=torch.float32)
            r4 = torch.tensor(self.roberta4[vid], dtype=torch.float32)
        else:
            sn = self.Sn[index].cpu()
            r1 = torch.tensor(self.roberta1[vid], dtype=torch.float32) * sn[:, 0].unsqueeze(dim=1)
            r2 = torch.tensor(self.roberta2[vid], dtype=torch.float32) * sn[:, 1].unsqueeze(dim=1)
            r3 = torch.tensor(self.roberta3[vid], dtype=torch.float32) * sn[:, 2].unsqueeze(dim=1)
            r4 = torch.tensor(self.roberta4[vid], dtype=torch.float32) * sn[:, 3].unsqueeze(dim=1)
        x = torch.cat((r1, r2, r3, r4, self.h[vid]), dim=1)

        return x, \
               torch.tensor([[1, 0] if x == '0' else [0, 1] for x in self.speakers[vid]], dtype=torch.float), \
               torch.tensor([1] * len(self.labels[vid]), dtype=torch.float), \
               torch.tensor(self.labels[vid], dtype=torch.long), \
               vid

    def __len__(self):
        return self.len

    def get_keys_lens(self):
        return self.keys_lens

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]

    def set_Sn(self, Sn):
        accum_item = [0]
        for i in self.lens:
            accum_item = accum_item + [accum_item[-1] + i]

        self.Sn = [Sn[accum_item[x]: accum_item[x + 1], :] for x in range(len(accum_item) - 1)]

    def set_h(self, H):
        accum_item = [0]
        for k, l in zip(self.keys, self.lens):
            accum_item = accum_item + [accum_item[-1] + l]

        for i in range(len(self.keys)):
            self.h[self.keys[i]] = H[accum_item[i]:accum_item[i + 1], :].detach().cpu()


class MELDDataset(Dataset):
    def __init__(self, path, dims, num_view, train=True):
        # if n_classes == 3:
        #     self.videoIDs, self.videoSpeakers, _, self.videoText, \
        #     self.videoAudio, self.videoSentence, self.trainVid, \
        #     self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        # elif n_classes == 7:
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoSentence, self.trainVid, \
        self.testVid, _ = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)
        self.lens = [len(self.videoLabels[x]) for x in self.keys]

        self.Sn = None
        self.num_view = num_view

        self.keys_lens = {}
        self.h = {}
        for k, l in zip(self.keys, self.lens):
            self.keys_lens[k] = l
            self.h[k] = torch.zeros((l, dims[-1]), dtype=torch.float)

    def __getitem__(self, index):
        vid = self.keys[index]
        if self.Sn is None:
            text = torch.tensor(self.videoText[vid], dtype=torch.float32)
            video = torch.tensor(self.videoAudio[vid], dtype=torch.float32)
        else:
            sn = self.Sn[index].cpu()
            text = torch.tensor(self.videoText[vid], dtype=torch.float32) * sn[:, 0].unsqueeze(dim=1)
            video = torch.tensor(self.videoAudio[vid], dtype=torch.float32) * sn[:, 1].unsqueeze(dim=1)
        x = torch.cat((text, video, self.h[vid]), dim=1)

        return x, \
               torch.tensor(self.videoSpeakers[vid], dtype=torch.float32), \
               torch.tensor([1] * len(self.videoLabels[vid]), dtype=torch.float32), \
               torch.tensor(self.videoLabels[vid], dtype=torch.long), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]

    def get_keys_lens(self):
        return self.keys_lens

    def set_Sn(self, Sn):
        accum_item = [0]
        for i in self.lens:
            accum_item = accum_item + [accum_item[-1] + i]

        self.Sn = [Sn[accum_item[x]: accum_item[x + 1], :] for x in range(len(accum_item) - 1)]

    def set_h(self, H):
        accum_item = [0]
        for k, l in zip(self.keys, self.lens):
            accum_item = accum_item + [accum_item[-1] + l]

        for i in range(len(self.keys)):
            self.h[self.keys[i]] = H[accum_item[i]:accum_item[i + 1], :].detach().cpu()


class IEMOCAPDataset(Dataset):
    def __init__(self, path, dims, num_view, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.lens = [len(self.videoLabels[x]) for x in self.keys]
        self.len = len(self.keys)
        self.num_view = num_view

        q_mask_ = []
        for vid in self.keys:
            q_mask = torch.tensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]],
                                  dtype=torch.float32)
            q_mask_.append(q_mask)
        self.q_mask_ = q_mask_

        self.u_mask_ = [torch.tensor([1] * x, dtype=torch.float32) for x in self.lens]
        self.label_ = [torch.tensor(self.videoLabels[x], dtype=torch.long) for x in self.keys]

        self.Sn = None

        self.keys_lens = {}
        self.h = {}
        self.imputation = {}
        for k, l in zip(self.keys, self.lens):
            self.keys_lens[k] = l
            self.h[k] = torch.zeros((l, dims[-1]), dtype=torch.float)

            imputation_ = {}
            for v in range(self.num_view):
                imputation_[str(v)] = torch.zeros((l, dims[v]), dtype=torch.float)
            self.imputation[k] = imputation_

    def __getitem__(self, index):
        vid = self.keys[index]
        if self.Sn is None:
            text = torch.tensor(self.videoText[vid], dtype=torch.float32)
            visual = torch.tensor(self.videoVisual[vid], dtype=torch.float32)
            video = torch.tensor(self.videoAudio[vid], dtype=torch.float32)
        else:
            sn = self.Sn[index].cpu()
            text = torch.tensor(self.videoText[vid], dtype=torch.float32) * sn[:, 0].unsqueeze(dim=1)
            visual = torch.tensor(self.videoVisual[vid], dtype=torch.float32) * sn[:, 1].unsqueeze(dim=1)
            video = torch.tensor(self.videoAudio[vid], dtype=torch.float32) * sn[:, 2].unsqueeze(dim=1)

        x = torch.cat((text, visual, video, self.h[vid]), dim=1)

        return x, \
               torch.tensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]], dtype=torch.float32), \
               torch.tensor([1] * len(self.videoLabels[vid]), dtype=torch.float32), \
               torch.tensor(self.videoLabels[vid], dtype=torch.long), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]

    def get_keys_lens(self):
        return self.keys_lens

    def set_Sn(self, Sn):
        accum_item = [0]
        for i in self.lens:
            accum_item = accum_item + [accum_item[-1] + i]

        self.Sn = [Sn[accum_item[x]: accum_item[x + 1], :] for x in range(len(accum_item) - 1)]

    def set_h(self, H):
        accum_item = [0]
        for k, l in zip(self.keys, self.lens):
            accum_item = accum_item + [accum_item[-1] + l]

        for i in range(len(self.keys)):
            self.h[self.keys[i]] = H[accum_item[i]:accum_item[i + 1], :].detach().cpu()


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
        self.context = None

        """
        this vid contains all long video names, for train:

        ['Ses04F_script02_1', 'Ses03F_impro03', 'Ses04F_script01_3', 'Ses01F_impro04', 'Ses01M_impro01',
         'Ses04M_impro01', 'Ses04F_impro08', 'Ses03M_impro08a', 'Ses02F_script03_1', 'Ses01M_impro04',
        ...
         'Ses01M_script03_1', 'Ses03F_impro04', 'Ses01M_impro02', 'Ses02F_script01_3', 'Ses04M_script03_1',
         'Ses01M_impro07']
         """
        self.vid = None
        self.get_vid()

        self.Sn = None
        self.dim_g = 512

    def get_data(self):
        text_view = torch.stack(self.text_, dim=0).cuda()
        visual_view = torch.stack(self.visual_, dim=0).cuda()
        audio_view = torch.stack(self.audio_, dim=0).cuda()
        if self.context is not None:
            # before set context information
            context = torch.cat(self.context, dim=0).cuda()
        else:
            context = torch.zeros((text_view.shape[0], self.dim_g), dtype=torch.float).cuda()

        return {"0": text_view, "1": visual_view, "2": audio_view, "3": context}

    def get_label(self):
        labels = torch.tensor(self.label_, dtype=torch.long).unsqueeze(1).cuda()
        return labels

    def get_vid(self):
        # remove repeated video ids and contain consequence in list
        vid = []
        for i in self.vid_:
            if not i in vid:
                vid.append(i)

        self.vid = vid

    def set_context(self, context):
        c_v = []
        for v in self.vid:
            c_v.append(context[v])
        self.context = c_v

    def set_Sn(self, Sn):
        self.Sn = Sn
        ones = torch.ones((self.Sn.shape[0], 1), dtype=torch.long).cuda()
        self.Sn = torch.cat((self.Sn, ones), dim=1)

    def get_Sn(self):
        return self.Sn

    def __len__(self):
        return self.len


class MELDDatasetUtter:
    def __init__(self, path, device, train=True):
        cache = pickle.load(open(path, 'rb'), encoding='latin1')
        if train:
            cache = cache["train"]
        else:
            cache = cache["test"]

        self.text_ = cache["text"]
        self.visual_ = cache["visual"]
        # self.audio_ = cache["audio"]
        self.label_ = cache["label"]
        self.vid_ = cache["vid"]
        self.idx_ = cache["idx"]
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.len = len(self.text_)
        self.device = device
        self.context = None

        self.vid = None
        self.get_vid()

        self.Sn = None
        self.dim_g = 512

    def get_data(self):
        text_view = torch.stack(self.text_, dim=0).cuda()
        visual_view = torch.stack(self.visual_, dim=0).cuda()
        # audio_view = torch.stack(self.audio_, dim=0).cuda()
        if self.context is not None:
            # before set context information
            context = torch.cat(self.context, dim=0).cuda()
        else:
            context = torch.zeros((text_view.shape[0], self.dim_g), dtype=torch.float).cuda()

        return {"0": text_view, "1": visual_view, "2": context}

    def get_label(self):
        labels = torch.tensor(self.label_, dtype=torch.long).unsqueeze(1).cuda()
        return labels

    def get_vid(self):
        # remove repeated video ids and contain consequence in list
        vid = []
        for i in self.vid_:
            if not i in vid:
                vid.append(i)

        self.vid = vid

    def set_context(self, context):
        c_v = []
        for v in self.vid:
            c_v.append(context[v])
        self.context = c_v

    def set_Sn(self, Sn):
        self.Sn = Sn
        ones = torch.ones((self.Sn.shape[0], 1), dtype=torch.long).cuda()
        self.Sn = torch.cat((self.Sn, ones), dim=1)

    def get_Sn(self):
        return self.Sn

    def __len__(self):
        return self.len


class EmoryNlpDatasetUtter:
    def __init__(self, path, device, train=True):
        cache = pickle.load(open(path, 'rb'), encoding='latin1')
        if train:
            cache = cache["train"]
        else:
            cache = cache["test"]

        self.r1_ = cache["r1"]
        self.r2_ = cache["r2"]
        self.r3_ = cache["r3"]
        self.r4_ = cache["r4"]
        # self.audio_ = cache["audio"]
        self.label_ = cache["label"]
        self.vid_ = cache["vid"]
        self.idx_ = cache["idx"]
        '''
        '''
        self.len = len(self.r1_)
        self.device = device
        self.context = None

        self.vid = None
        self.get_vid()

        self.Sn = None
        self.dim_g = 512

    def get_data(self):
        r1_view = torch.stack(self.r1_, dim=0).cuda()
        r2_view = torch.stack(self.r2_, dim=0).cuda()
        r3_view = torch.stack(self.r3_, dim=0).cuda()
        r4_view = torch.stack(self.r4_, dim=0).cuda()
        # audio_view = torch.stack(self.audio_, dim=0).cuda()
        if self.context is not None:
            # before set context information
            context = torch.cat(self.context, dim=0).cuda()
        else:
            context = torch.zeros((r1_view.shape[0], self.dim_g), dtype=torch.float).cuda()

        return {"0": r1_view, "1": r2_view, "2": r3_view, "3": r4_view, "4": context}

    def get_label(self):
        labels = torch.tensor(self.label_, dtype=torch.long).unsqueeze(1).cuda()
        return labels

    def get_vid(self):
        # remove repeated video ids and contain consequence in list
        vid = []
        for i in self.vid_:
            if not i in vid:
                vid.append(i)

        self.vid = vid

    def set_context(self, context):
        c_v = []
        for v in self.vid:
            c_v.append(context[v])
        self.context = c_v

    def set_Sn(self, Sn):
        self.Sn = Sn
        ones = torch.ones((self.Sn.shape[0], 1), dtype=torch.long).cuda()
        self.Sn = torch.cat((self.Sn, ones), dim=1)

    def get_Sn(self):
        return self.Sn

    def __len__(self):
        return self.len


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


class HDataset(Dataset):
    def __init__(self, H, q_mask, u_mask, label, keys_lens):
        self.data = H
        self.q_mask = q_mask
        self.u_mask = u_mask
        self.label = label
        accum_item = [0]
        for k, l in keys_lens.items():
            accum_item = accum_item + [accum_item[-1] + l]
        self.accum_item = accum_item

        self.len = len(accum_item) - 1

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.data[self.accum_item[item]:self.accum_item[item + 1], :], \
               self.q_mask[item], self.u_mask[item], self.label[item]

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[0]), pad_sequence(dat[1]), pad_sequence(dat[2], True), pad_sequence(dat[3], True)]


if __name__ == "__main__":
    dataset_train = EmoryNlpDataset(path=args.data_path, dims=args.dim_features, num_view=args.num_views)
    dataset_test = EmoryNlpDataset(path=args.data_path, dims=args.dim_features, num_view=args.num_views, train=False)
    generate_utterance('./data/EMORY_utterance_raw.pkl', dataset_train, dataset_test)
    # d_T = 100, d_V=512, d_A = 100
    pass

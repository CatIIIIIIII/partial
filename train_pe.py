import time

from arg_setting import args
from torch.utils.data import DataLoader
from data_loader import IEMOCAPDataset, IEMOCAPDatasetUtter, get_loaders, HDataset
from cpm import CPMNet_Works
from dialogue import Dialogue_Works
from utils import get_sn, ave
import numpy as np
import torch
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    use_p = args.use_p
    epochs_ep = args.epochs_ep
    num_views = args.num_views
    data_path = args.data_path
    e_batch_size = args.e_batch_size
    missing_rate = 0.8

    dim_features = args.dim_features
    # dimension of hidden representation from different modalities
    dim_init = sum(dim_features)
    dim_h = args.dim_h
    lr_p = args.lr_p
    lambda_p = args.lambda_p
    n_classes = args.n_classes
    device = torch.device(args.device)
    context_attention = args.context_attention
    lr_e = args.lr_e
    loss_weights = args.loss_weights
    model_type = args.model_type

    dim_g = args.dim_g
    dim_p = args.dim_p
    dim_e = args.dim_e
    dim_y = args.dim_y
    dim_a = args.dim_a

    epochs_init = args.epochs_init

    rec_dropout = args.rec_dropout
    dropout = args.dropout
    epochs_e = args.epochs_e
    steps_e = args.steps_e

    # Load dataset of emotion scenario (long video of dialogue)
    train_set = IEMOCAPDataset(path=data_path, dim_h=dim_h)
    test_set = IEMOCAPDataset(path=data_path, dim_h=dim_h, train=False)
    # video ids and utter lens of train/test sets
    train_keys_lens = {}
    test_keys_lens = {}
    for k, l in zip(train_set.keys, train_set.lens):
        train_keys_lens[k] = l
    for k, l in zip(test_set.keys, test_set.lens):
        test_keys_lens[k] = l

    # Load dataset of emotion clips (short video of utterance)
    train_set_utter = IEMOCAPDatasetUtter(args.utterance_path, device)
    train_data_utter = train_set_utter.get_data()
    train_gt_utter = train_set_utter.get_label()
    # train
    test_set_utter = IEMOCAPDatasetUtter(args.utterance_path, device, train=False)
    test_data_utter = test_set_utter.get_data()
    test_gt_utter = test_set_utter.get_label()

    # Randomly generated missing matrix
    len_train_utter = len(train_set_utter)
    len_test_utter = len(test_set_utter)

    Sn = get_sn(num_views, len_train_utter + len_test_utter, missing_rate)  # [num_samples, num_views]
    Sn_train = Sn[np.arange(len_train_utter)]
    Sn_test = Sn[np.arange(len_test_utter) + len_train_utter]

    Sn = torch.tensor(Sn, dtype=torch.long).to(device)
    Sn_train = torch.tensor(Sn_train, dtype=torch.long).to(device)
    Sn_test = torch.tensor(Sn_test, dtype=torch.long).to(device)

    train_set_utter.set_Sn(Sn_train)
    test_set_utter.set_Sn(Sn_test)

    train_1hot = (torch.zeros([len_train_utter, n_classes]).to(device).scatter_(1, train_gt_utter, 1))

    # Model building
    model_p = CPMNet_Works(num_views + 1,  # number of view and context
                           len(train_set_utter),
                           len(test_set_utter),
                           dim_features + [dim_g],
                           dim_h,
                           lr_p,
                           lambda_p).to(device)

    data_set_e_train = train_set
    data_set_e_test = test_set

    steps_p = args.steps_p
    epochs_p = args.epochs_p

    H_train = model_p.train_model(train_set_utter.get_data(),
                                  train_set_utter.get_Sn(),
                                  train_1hot,
                                  train_gt_utter,
                                  epochs_p[0],
                                  steps_p)
    # ----- test partial multi-view ----- #
    H_test = model_p.test_model(test_set_utter.get_data(),
                                test_set_utter.get_Sn(),
                                epochs_p[1])
    label_pre = ave(H_train, H_test, train_1hot)
    print('Accuracy on the test set is {:.4f}'.format(accuracy_score(test_gt_utter.cpu(), label_pre)))

from arg_setting import args
from torch.utils.data import DataLoader
from data_loader import IEMOCAPDataset, IEMOCAPDatasetUtter, get_loaders
from cpm import CPMNet_Works
from utils import get_sn, ave
import numpy as np
import torch
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    epochs_ep = args.epochs_ep
    num_views = args.num_views
    data_path = args.data_path
    e_batch_size = args.e_batch_size
    missing_rate = args.missing_rate
    steps_p = args.steps_p
    epochs_p = args.epochs_p
    # dimension of different modalities
    dim_features = args.dim_features
    dim_h = args.dim_h
    lr_p = args.lr_p
    lambda_p = args.lambda_p
    class_num = args.n_classes
    device = torch.device(args.device)

    # Load dataset of emotion scenario (long video of dialogue)
    train_set, test_set = IEMOCAPDataset(path=data_path), IEMOCAPDataset(path=data_path, train=False)
    # video ids of train/test sets
    train_keys, test_keys = train_set.keys, test_set.keys
    train_lens, test_lens = train_set.lens, test_set.lens

    # Load dataset of emotion clips (short video of utterance)
    train_set_utter = IEMOCAPDatasetUtter(args.utterance_path, device)
    train_data_utter = train_set_utter.get_data()
    train_gt_utter = train_set_utter.get_label()

    test_set_utter = IEMOCAPDatasetUtter(args.utterance_path, device, train=False)
    test_data_utter = test_set_utter.get_data()
    test_gt_utter = test_set_utter.get_label()

    # Randomly generated missing matrix
    len_train_utter = len(train_set_utter)
    len_test_utter = len(test_set_utter)
    Sn = get_sn(num_views, len_train_utter + len_test_utter, args.missing_rate)
    Sn_train = Sn[np.arange(len_train_utter)]
    Sn_test = Sn[np.arange(len_test_utter) + len_train_utter]

    Sn = torch.tensor(Sn, dtype=torch.long).to(device)
    Sn_train = torch.tensor(Sn_train, dtype=torch.long).to(device)
    Sn_test = torch.tensor(Sn_test, dtype=torch.long).to(device)

    # Model building
    model_p = CPMNet_Works(num_views,
                           len(train_set_utter),
                           len(test_set_utter),
                           dim_features,
                           dim_h,
                           lr_p,
                           lambda_p).to(device)

    train_1hot = (torch.zeros(len_train_utter, class_num).to(device).scatter_(1, train_gt_utter, 1))
    # print(train_data_utter["0"].shape)
    # print(train_data_utter["1"].shape)
    # print(train_data_utter["2"].shape)

    H_train = model_p.train_model(train_data_utter,
                                  Sn_train,
                                  train_1hot,
                                  train_gt_utter,
                                  epochs_p[0],
                                  steps_p)

    # test
    H_test = model_p.test_model(test_data_utter,
                                Sn_test,
                                epochs_p[1])

    label_pre = ave(H_train, H_test, train_1hot.cuda())
    print('Accuracy on the test set is {:.4f}'
          .format(accuracy_score(test_gt_utter.cpu().numpy(), label_pre)))

    for e_ep in range(epochs_ep):
        # ----- train partial multi-view ----- #
        pass

        # ----- train emotion flow ----- #
        pass

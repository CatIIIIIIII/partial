from arg_setting import args
from data_loader import IEMOCAPDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import numpy as np
import torch
from arg_setting import args
from models import Model, MaskedNLLLoss
import torch.optim as optim
from pathlib import Path
import time


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer is not None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        # import ipdb;ipdb.set_trace()
        textf, visuf, acouf, qmask, umask, label = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        # log_prob = model(torch.cat((textf, acouf, visuf), dim=-1), qmask, umask, att2=True)  # seq_len, batch, n_classes
        log_prob = model(textf, qmask, umask, att2=True)  # seq_len, batch, n_classes
        # log_prob, alpha, alpha_f, alpha_b = model(textf, qmask, umask, att2=True)  # seq_len, batch, n_classes

        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            # if args.tensorboard:
            #     for param in model.named_parameters():
            #         writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        # else:
        #     alphas += alpha
        #     alphas_f += alpha_f
        #     alphas_b += alpha_b
        #     vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


if __name__ == "__main__":
    args.cuda = torch.cuda.is_available() and args.cuda is True

    batch_size = args.batch_size
    n_classes = 6
    cuda = args.cuda
    n_epochs = args.epochs

    D_m = 100
    D_g = 500
    D_p = 500
    D_e = 300
    D_h = 300

    D_a = 100  # concat attention

    model = Model(D_m, D_g, D_p, D_e, D_h,
                  n_classes=n_classes,
                  context_attention=args.attention,
                  dropout_rec=args.rec_dropout,
                  dropout=args.dropout)
    if cuda:
        model.cuda()

    loss_weights = torch.tensor([
        1 / 0.086747,
        1 / 0.144406,
        1 / 0.227883,
        1 / 0.160585,
        1 / 0.127711,
        1 / 0.252668,
    ], dtype=torch.float32)

    if args.class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    data_path = Path(args.data_root) / (args.dataset + "_features_raw.pkl")

    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(data_path,
                                                                  valid=0.0,
                                                                  batch_size=batch_size,
                                                                  num_workers=2)

    best_loss, best_label, best_pred, best_mask = None, None, None, None
    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function,
                                                                              train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, val_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)

        test_loss, test_acc, test_label, test_pred, \
        test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss is None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_attn = \
                test_loss, test_label, test_pred, test_mask, attentions

        print('epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore{} '
              'test_loss {} test_acc {} test_fscore {} time {}'.
              format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,
                     test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))

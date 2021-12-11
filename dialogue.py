from models import Model, MaskedNLLLoss
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


class Dialogue_Works:
    def __init__(self, model, D_h, D_g, D_p, D_e, D_y,
                 n_classes,
                 context_attention,
                 dropout_rec,
                 dropout,
                 lr,
                 loss_weights=None):
        """
        :param model: model type of emotional state
        :param D_h: dimension of input
        :param D_g: dimension of global state
        :param D_p: dimension of personal state
        :param D_e: dimension of emotional state
        :param D_y: dimension of classifier state
        :param n_classes: number of emotional classes
        :param context_attention: attention type
        :param dropout_rec:
        :param dropout:
        :param lr: learning rate
        :param loss_weights: class balanced weights
        """
        self.model = model
        self.D_h, self.D_g, self.D_p, self.D_e, self.D_y = D_h, D_g, D_p, D_e, D_y
        self.n_classes = n_classes
        self.context_attention = context_attention
        self.dropout_rec = dropout_rec
        self.dropout = dropout
        self.loss_weights = loss_weights
        self.lr = lr

        self.net, self.loss_function, self.optimizer = self.build_model()

    def train_test_model(self, data_loader, step, keys_lens, train=True):
        for e in range(step):
            losses = []
            preds = []
            labels = []
            masks = []
            alphas, alphas_f, alphas_b, vids = [], [], [], []
            if train:
                self.net.train()
            else:
                self.net.eval()

            context = {}
            for data in data_loader:
                if train:
                    self.optimizer.zero_grad()

                x, q_mask, u_mask, label = [d.cuda() for d in data[:-1]]
                vid = data[-1]
                # print(x.shape)
                # print(q_mask.shape)
                # print(u_mask.shape)
                # print(label.shape)
                log_prob, c = self.net(x, q_mask, u_mask, att2=True)  # seq_len, batch, n_classes

                c = c.detach()
                for i in range(len(vid)):
                    key = vid[i]
                    context[key] = c[:keys_lens[key], i, :]

                lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
                labels_ = label.view(-1)  # batch*seq_len
                loss = self.loss_function(lp_, labels_, u_mask)

                pred_ = torch.argmax(lp_, 1)  # batch*seq_len
                preds.append(pred_.data.cpu().numpy())
                labels.append(labels_.data.cpu().numpy())
                masks.append(u_mask.view(-1).cpu().numpy())

                losses.append(loss.item() * masks[-1].sum())

                if train:
                    loss.backward()
                    self.optimizer.step()

                # if args.tensorboard:
                #     for param in model.named_parameters():
                #         writer.add_histogram(param[0], param[1].grad, epoch)
                # else:
                #     alphas += alpha
                #     alphas_f += alpha_f
                #     alphas_b += alpha_b
                #     vids += data[-1]

            if preds:
                preds = np.concatenate(preds)
                labels = np.concatenate(labels)
                masks = np.concatenate(masks)
            else:
                return float('nan'), float('nan'), [], [], [], float('nan'), []

            avg_loss = round(np.sum(losses) / np.sum(masks), 4)
            avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
            avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

            return avg_loss, avg_accuracy, avg_fscore, context
            # output = 'train: ' if train else 'test: '
            # output += 'epoch {} avg_loss {} avg_accuracy {} avg_fscore {}'.format(e, avg_loss, avg_accuracy, avg_fscore)
            # print(output)
            # return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]

    def test_model(self):
        pass

    def build_model(self):
        net = None
        if self.model == "base":
            net = Model(self.D_h, self.D_g, self.D_p, self.D_e, self.D_y,
                        n_classes=self.n_classes,
                        context_attention=self.context_attention,
                        dropout_rec=self.dropout_rec,
                        dropout=self.dropout).cuda()

        loss_function = None
        if self.loss_weights is not None:
            loss_function = MaskedNLLLoss(self.loss_weights.cuda())

        optimizer = optim.Adam(net.parameters(),
                               lr=self.lr[0],
                               weight_decay=self.lr[1])

        return net, loss_function, optimizer


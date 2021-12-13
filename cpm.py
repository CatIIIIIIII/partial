import time

import torch
from utils import xavier_init
from models import CPMNets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# ----- Partial Multi-view Network Works ----- #
class CPMNet_Works(nn.Module):  # Main parts of the test code
    """build model
    """

    def __init__(self, view_num, trainLen, testLen, dim_feats, lsd_dim, lr, lamb):
        """
        :param lr:learning rate of network and h
        :param view_num:view number
        :param dim_feats:dimension of all input features
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        super(CPMNet_Works, self).__init__()
        # initialize parameter
        if lr is None:
            lr = [0.001, 0.001]
        self.view_num = view_num
        layer_size = [[128, i] for i in dim_feats]
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        self.lr = lr
        # initialize latent space data
        self.h_train = self.H_init('train')
        self.h_test = self.H_init('test')
        self.h = torch.cat((self.h_train, self.h_test), 0).cuda()
        # initialize nets for different views
        self.net, self.train_net_op = self.build_model()

    def H_init(self, tvt):
        h = None
        if tvt == 'train':
            h = Variable(xavier_init(self.trainLen, self.lsd_dim), requires_grad=True)
        elif tvt == 'test':
            h = Variable(xavier_init(self.testLen, self.lsd_dim), requires_grad=True)
        return h

    def reconstruction_loss(self, h, x, sn):
        loss = 0
        x_pred = self.calculate(h)
        for num in range(self.view_num):
            loss = loss + (torch.pow((x_pred[str(num)].cpu() - x[str(num)].cpu()), 2) * sn[str(num)].cpu()).sum()
        return loss

    def classification_loss(self, label_1hot, gt, h_temp):
        h_temp = h_temp.float()
        h_temp = h_temp.cuda()
        F_h_h = torch.mm(h_temp, h_temp.T)
        F_hn_hn = torch.eye(F_h_h.shape[0], F_h_h.shape[1])
        F_h_h = F_h_h - F_h_h * (F_hn_hn.cuda())
        # should sub 1.Avoid numerical errors; the number of samples of per label
        label_num = label_1hot.sum(0, keepdim=True)
        label_1hot = label_1hot.float()
        F_h_h_sum = torch.mm(F_h_h, label_1hot)
        F_h_h_mean = F_h_h_sum / label_num

        gt1 = torch.max(F_h_h_mean, dim=1)[1]  # gt begin from 1
        gt_ = gt1.type(torch.IntTensor) + 1
        F_h_h_mean_max = torch.max(F_h_h_mean, dim=1, keepdim=False)[0]
        gt_ = gt_.cuda()
        gt_ = gt_.reshape([gt_.shape[0], 1])
        theta = torch.ne(gt, gt_).type(torch.float)
        F_h_hn_mean_ = F_h_h_mean * label_1hot
        F_h_hn_mean = F_h_hn_mean_.sum(dim=1)
        F_h_h_mean_max = F_h_h_mean_max.reshape([F_h_h_mean_max.shape[0], 1])
        F_h_hn_mean = F_h_hn_mean.reshape([F_h_hn_mean.shape[0], 1])
        theta = theta.cuda()

        return (torch.nn.functional.relu(theta + F_h_h_mean_max - F_h_hn_mean)).sum()

    def train_model(self, data, sn, label_1hot, gt, epoch, step):
        global Reconstruction_LOSS
        sn1 = dict()
        for i in range(self.view_num):
            sn1[str(i)] = sn[:, i].reshape(self.trainLen, 1)
        for e in range(epoch):
            st = time.time()
            for i in range(step[0]):
                Reconstruction_LOSS = self.reconstruction_loss(self.h_train, data, sn1).float().cuda()
                for v_num in range(self.view_num):
                    self.train_net_op[v_num].zero_grad()
                Reconstruction_LOSS.backward()

                for v_num in range(self.view_num):
                    self.train_net_op[v_num].step()

            train_hn_op = torch.optim.Adam([self.h_train], self.lr[1])
            for i in range(step[1]):
                loss = (self.reconstruction_loss(self.h_train, data, sn1) +
                        self.lamb * self.classification_loss(label_1hot, gt, self.h_train)).float().cuda()
                train_hn_op.zero_grad()
                loss.backward()
                train_hn_op.step()
            Classification_LOSS = self.classification_loss(label_1hot, gt, self.h_train)
            Reconstruction_LOSS = self.reconstruction_loss(self.h_train, data, sn1)
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.2f}, Classification Loss = {:.2f}, Time = {:.2f} " \
                .format((e + 1), Reconstruction_LOSS, Classification_LOSS, time.time() - st)
            print(output)
        return self.h_train

    def build_model(self):
        # initialize network
        net = dict()
        train_net_op = []
        for v_num in range(self.view_num):
            net[str(v_num)] = CPMNets(self.view_num, self.trainLen, self.testLen, self.layer_size, v_num,
                                      self.lsd_dim, self.lamb).cuda()
            train_net_op.append(torch.optim.Adam([{"params": net[str(v_num)].parameters()}], self.lr[0]))
        return net, train_net_op

    def calculate(self, h):
        h_views = dict()
        for v_num in range(self.view_num):
            h_views[str(v_num)] = self.net[str(v_num)](h)
        return h_views

    def test_model(self, data, sn, epoch):
        sn1 = dict()
        for i in range(self.view_num):
            sn1[str(i)] = sn[:, i].reshape(self.testLen, 1).cuda()
        adj_hn_op = torch.optim.Adam([self.h_test], self.lr[0])
        for e in range(epoch):
            # update the h
            for i in range(5):
                Reconstruction_LOSS = self.reconstruction_loss(self.h_test, data, sn1).float()
                adj_hn_op.zero_grad()
                Reconstruction_LOSS.backward()
                adj_hn_op.step()
            Reconstruction_LOSS = self.reconstruction_loss(self.h_test, data, sn1).float()
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.2f}".format((e + 1), Reconstruction_LOSS)
            print(output)
        return self.h_test

    # def init(self):
    #     init_parameters()

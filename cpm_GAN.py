import time
from pprint import pprint

import torch
import torch.nn as nn

from models import CpmGenerator, CpmDiscriminator
from utils import xavier_init
import torch.autograd as autograd


# ----- Partial Multi-view Network Works ----- #
class CPMNet_Works_GAN(nn.Module):  # Main parts of the test code
    """
    build model
    """

    def __init__(self, view_num, trainLen, testLen, dim_feats, lsd_dim, lr, lamb, p_batch_size):
        """
        :param lr:learning rate of network and h
        :param view_num:view number
        :param dim_feats:dimension of all input features
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        super(CPMNet_Works_GAN, self).__init__()
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
        self.h = torch.cat((self.h_train, self.h_test), dim=0).cuda()
        # initialize nets for different views
        self.g, self.d, self.g_optim, self.d_optim, self.h_optim = self.build_model()
        self.batch_size = p_batch_size

    def H_init(self, tvt):
        h = None
        if tvt == 'train':
            h = xavier_init(self.trainLen, self.lsd_dim).cuda().detach().requires_grad_(True)
        elif tvt == 'test':
            h = xavier_init(self.testLen, self.lsd_dim).cuda().detach().requires_grad_(True)
        return h

    def reconstruction_loss(self, h, x, sn):
        loss = torch.zeros([1]).cuda()
        x_pred = self.calculate(h)
        for v in range(self.view_num):
            loss = loss + torch.sum(torch.pow((x_pred[str(v)] - x[str(v)]), 2) * sn[str(v)])
        return loss

    def adversarial_loss(self, hI_, xI):
        loss = torch.zeros([1]).cuda()
        for v in range(self.view_num - 1):
            h_v = hI_[str(v)]
            # print(h_v.shape)

            xI_v = xI[str(v)]
            # positive samples
            xI_v_ = torch.repeat_interleave(xI_v, h_v.shape[0], dim=0)

            # negative samples
            h_v_ = torch.repeat_interleave(h_v, xI_v.shape[0], dim=0)
            fake_x = self.g[str(v)](h_v_)

            real_validity = self.d[str(v)](xI_v_)
            fake_validity = self.d[str(v)](fake_x)

            # Compute W-div gradient penalty
            real_grad_out = torch.ones([xI_v_.shape[0], 1], requires_grad=False).cuda()
            real_grad = autograd.grad(
                real_validity, xI_v_, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
            real_grad_norm = real_grad.view(real_grad.shape[0], -1).pow(2).sum(1) ** (6 / 2)

            fake_grad_out = torch.ones([fake_x.shape[0], 1], requires_grad=False).cuda()
            fake_grad = autograd.grad(
                fake_validity, fake_x, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
            fake_grad_norm = fake_grad.view(fake_grad.shape[0], -1).pow(2).sum(1) ** (6 / 2)

            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * 2 / 2

            # Adversarial loss
            loss += -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

        return loss

    def generator_loss(self, hI_):
        loss = torch.zeros([1]).cuda()
        for v in range(self.view_num - 1):
            h_v = hI_[str(v)]
            fake_x = self.g[str(v)](h_v)
            fake_validity = self.d[str(v)](fake_x)
            loss += -torch.mean(fake_validity)
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

        for e in range(epoch):
            # ===== data for adversarial loss ===== #
            sn_v, data_v, xI_v, hI_ = dict(), dict(), dict(), dict()
            for v in range(self.view_num):
                sn_v[str(v)] = sn[:, v].reshape(self.trainLen, 1)

                # data_v[str(v)] = data[str(v)]
                # positive samples
                xI = data[str(v)][torch.nonzero(sn_v[str(v)].squeeze(), as_tuple=False)].squeeze(dim=1)
                xI.requires_grad = True

                idx = torch.randperm(xI.shape[0])
                xI_sub = xI[idx[:int(1*xI.shape[0])]]
                # print(xI_sub.shape)
                xI_v[str(v)] = xI

                # negative samples
                hI_[str(v)] = self.h_train[torch.nonzero(1 - sn_v[str(v)].squeeze(), as_tuple=False)].squeeze(dim=1)

            st = time.time()
            i_g = 0
            loss_d, loss_g, loss_rec, loss_cls = torch.zeros([0]), torch.zeros([0]), torch.zeros([0]), torch.zeros([0])
            batch_index = range(0, xI_v['0'].shape[0], self.batch_size)
            for i in range(len(batch_index)-1):
                i_g += 1
                xI_batch = {'0': xI_v['0'][batch_index[i]:batch_index[i+1], :],
                            '1': xI_v['1'][batch_index[i]:batch_index[i+1], :],
                            '2': xI_v['2'][batch_index[i]:batch_index[i+1], :],
                            '3': xI_v['3'][batch_index[i]:batch_index[i+1], :]}
                # ===== compute loss ===== #
                # reconstruction
                loss_rec = self.reconstruction_loss(self.h_train, data, sn_v)
                # adversarial
                loss_d = self.adversarial_loss(hI_, xI_batch)

                loss = loss_rec + loss_d
                # ===== update networks ===== #
                # update discriminator
                [self.d_optim[v].zero_grad() for v in range(self.view_num)]
                loss.backward(retain_graph=True)
                [self.d_optim[v].step() for v in range(self.view_num)]
                # update generator
                if i_g % 2 == 0:
                    loss_rec = self.reconstruction_loss(self.h_train, data, sn_v)
                    loss_g = self.generator_loss(hI_)
                    loss = loss_rec + loss_g

                    [self.g_optim[v].zero_grad() for v in range(self.view_num)]
                    loss.backward(retain_graph=True)
                    [self.g_optim[v].step() for v in range(self.view_num)]
                # update h
                loss_cls = self.classification_loss(label_1hot, gt, self.h_train)
                loss_rec = self.reconstruction_loss(self.h_train, data, sn_v)
                loss_d = self.adversarial_loss(hI_, xI_batch)
                loss = loss_d + loss_rec + loss_cls
                self.h_optim.zero_grad()
                loss.backward(retain_graph=True)
                self.h_optim.step()

            output = "Epoch : {:.0f}  ===> Adversarial Loss = {:.2f}, Generator Loss = {:.2f}, " \
                     "Reconstruction Loss = {:.2f}, Classification Loss = {:.2f}, Time = {:.2f} " \
                .format((e + 1), loss_d.item(), loss_g.item(), loss_rec.item(), loss_cls.item(), time.time() - st)
            print(output)

        return self.h_train

    def build_model(self):
        # initialize network
        g, d = dict(), dict()
        g_optim, d_optim = [], []
        for v in range(self.view_num):
            g[str(v)] = nn.DataParallel(CpmGenerator(self.layer_size[v], self.lsd_dim))
            d[str(v)] = nn.DataParallel(CpmDiscriminator(self.layer_size[v]))
            g_optim.append(torch.optim.Adam([{"params": g[str(v)].parameters()}], self.lr[0], betas=(0.5, 0.999)))
            d_optim.append(torch.optim.Adam([{"params": d[str(v)].parameters()}], self.lr[0], betas=(0.5, 0.999)))
        h_optim = torch.optim.Adam([self.h_train], self.lr[0], betas=(0.5, 0.999))

        return g, d, g_optim, d_optim, h_optim

    def calculate(self, h):
        h_views = dict()
        for v in range(self.view_num):
            h_views[str(v)] = self.g[str(v)](h)
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
            Reconstruction_LOSS = self.reconstruction_loss(self.h_test, data, sn1).item()
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.2f}".format((e + 1), Reconstruction_LOSS)
            print(output)
        return self.h_test

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_parameters(net):
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.xavier_normal_(param)
        elif 'bias' in name:
            init.constant_(param, val=0)


# ----- Dialogue Emotion Networks ----- #
class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M):
        """
        M -> (seq_len, batch, vector)
        """
        scale = self.scalar(M)  # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, vector

        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim is not None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            # torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim), g_hist..., key and value
        x -> (batch, cand_dim), U..., query
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            print(M.shape)
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_)) * mask.unsqueeze(1), dim=2)  # batch, 1, seqlen
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
            # import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim

        return attn_pool, alpha


class DialogueRNNCell(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.g_cell = nn.GRUCell(D_m + D_p, D_g)
        self.p_cell = nn.GRUCell(D_m + D_g, D_p)
        self.e_cell = nn.GRUCell(D_p, D_e)

        self.dropout = nn.Dropout(dropout)

        if context_attention == 'simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, 0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        q0_sel -> batch, D_p
        U_c_ -> batch, party, D_m + D_a
        """
        qm_idx = torch.argmax(qmask, 1)  # indicate which person
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U, q0_sel], dim=1),
                         torch.zeros((U.size()[0], self.D_g), dtype=torch.float32).type(U.type()) if g_hist.size()[
                                                                                                         0] == 0 else
                         g_hist[-1])
        g_ = self.dropout(g_)
        if g_hist.size()[0] == 0:
            c_ = torch.zeros((U.size()[0], self.D_g), dtype=torch.float32).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist, U)  # batch_size, D_c

        # c_ = torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0\
        #         else self.attention(g_hist,U)[0] # batch, D_g
        U_c_ = torch.cat((U, c_), dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)

        qs_ = self.p_cell(U_c_.contiguous().view(-1, self.D_m + self.D_g),
                          q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
        qs_ = self.dropout(qs_)

        ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_ * (1 - qmask_) + qs_ * qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0] == 0 else e0
        e_ = self.e_cell(self._select_parties(q_, qm_idx), e0)
        e_ = self.dropout(e_)
        return g_, q_, e_, c_.detach(), alpha


class DialogueRNNCell_test(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, party,
                 context_attention='simple', party_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNCell_test, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.party = party

        self.g_cell = nn.GRUCell(D_m + D_p, D_g)
        self.p_cell = nn.GRUCell(D_m + D_g, D_p)
        self.e_cell = nn.GRUCell(D_p, D_e)
        self.dropout = nn.Dropout(dropout)

        if context_attention == 'simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

        self.attention_p = MatchingAttention(D_p, D_e, D_a, party_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, 0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, q_hist, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q_hist -> t-1, batch, party, D_p
        Q -> batch, party, D_p
        q0 -> batch, party, D_p
        e0 -> batch, party, D_e
        q0_sel -> batch, D_p
        U_c_ -> batch, party, D_m + D_g
        """
        qm_idx = torch.argmax(qmask, 1)  # indicate which person
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U, q0_sel], dim=1),
                         torch.zeros((U.size()[0], self.D_g), dtype=torch.float32).type(U.type()) if g_hist.size()[
                                                                                                         0] == 0 else
                         g_hist[-1])
        g_ = self.dropout(g_)
        if g_hist.size()[0] == 0:
            c_ = torch.zeros((U.size()[0], self.D_g), dtype=torch.float32).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist, U)  # batch_size, D_g

        # c_ = torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0\
        #         else self.attention(g_hist,U)[0] # batch, D_g
        U_c_ = torch.cat((U, c_), dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)

        qs_ = self.p_cell(U_c_.contiguous().view(-1, self.D_m + self.D_g),
                          q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
        qs_ = self.dropout(qs_)

        ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_ * (1 - qmask_) + qs_ * qmask_

        # personal attention for emotion context
        Q = torch.zeros((U.shape[0], self.party, self.D_p), dtype=torch.float32).type(U.type())
        if q_hist.size()[0] == 0:
            # batch, party, D_p
            alpha_p = None
        else:
            for p in range(self.party):
                Q_, _ = self.attention_p(q_hist[:, :, p, :], e0[:, 1-p, :])  # batch_size, D_p
                Q[:, p, :] = Q_

        e_ = self.e_cell(Q.contiguous().view(-1, self.D_p),
                          e0.view(-1, self.D_e)).view(U.size()[0], -1, self.D_e)
        e_ = self.dropout(e_)

        # e_ = self.e_cell(self._select_parties(q_, qm_idx), e0)
        e_ = self.dropout(e_)
        print(e_.shape)
        return g_, q_, e_, c_.detach(), alpha


class DialogueRNN(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, party,
                 context_attention='simple', party_attention=None, D_a=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)
        self.party_attention = party_attention

        # self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e, context_attention, D_a, dropout)
        if party_attention is not None:
            self.dialogue_cell = DialogueRNNCell_test(D_m, D_g, D_p, D_e, party,
                                                      context_attention, party_attention, D_a, dropout)
        else:
            self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e, context_attention, D_a, dropout)

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        e -> seq_len, batch, party, D_e
        c -> # seq_len, batch, D_e
        """

        g_hist = torch.zeros(0).type(U.type())  # 0-dimensional tensor
        q_hist = torch.zeros(0).type(U.type())  # 0-dimensional tensor

        q_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_p).type(U.type())  # batch, party, D_p
        e_ = torch.zeros(1, qmask.size()[1], qmask.size()[2], self.D_e).type(U.type())  # 1, batch, party, D_e
        e = e_
        c_ = torch.zeros(0).type(U.type()).cuda()
        c = c_

        alpha = []
        for u_, qmask_ in zip(U, qmask):
            g_, q_, e_, c_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, q_hist, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)], 0)
            q_hist = torch.cat([q_hist, q_.unsqueeze(0)], 0)

            e = torch.cat((e, e_.unsqueeze(0)), dim=0)
            c = torch.cat((c, c_.unsqueeze(0)), dim=0)
            if type(alpha_) != type(None):
                alpha.append(alpha_[:, 0, :])

        return e, c, alpha


class Model(nn.Module):
    def __init__(self, D_h, D_g, D_p, D_e, D_y, party,
                 n_classes, context_attention='simple', party_attention='simple',
                 D_a=100, dropout_rec=0.5, dropout=0.5):
        super(Model, self).__init__()

        self.D_h = D_h
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_y = D_y
        self.party = party
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        # self.dropout_rec = nn.Dropout(0.2)
        self.dropout_rec = nn.Dropout(dropout + 0.15)
        self.dialog_rnn = DialogueRNN(D_h, D_g, D_p, D_e, self.party,
                                      context_attention, party_attention, D_a, dropout_rec)
        self.linear1 = nn.Linear(D_e, D_y)
        # self.linear2     = nn.Linear(D_h, D_h)
        # self.linear3     = nn.Linear(D_h, D_h)
        self.smax_fc = nn.Linear(D_y, n_classes)

        self.matchatt = MatchingAttention(D_e, D_e, att_type='general2')

    def forward(self, U, qmask, umask=None, att2=False):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions, c, _ = self.dialog_rnn(U, qmask)  # seq_len, batch, D_e
        # print(emotions)
        emotions = self.dropout_rec(emotions)

        # emotions = emotions.unsqueeze(1)
        if att2:
            att_emotions = []
            for t in emotions:
                att_emotions.append(self.matchatt(emotions, t, mask=umask)[0].unsqueeze(0))
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear1(att_emotions))
        else:
            hidden = F.relu(self.linear1(emotions))
        # hidden = F.relu(self.linear2(hidden))
        # hidden = F.relu(self.linear3(hidden))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)  # seq_len, batch, n_classes
        return log_prob, c


class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


# ----- Partial Multi-view Networks ----- #
class CPMNets(nn.Module):  # The architecture of the CPM
    """build model
    """

    def __init__(self, view_num, trainLen, testLen, layer_size, v, lsd_dim=128, lamb=1):
        """
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        super(CPMNets, self).__init__()
        # initialize parameter
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        # initialize forward methods
        self.net = self._make_view(v).cuda()

    def forward(self, h):
        h_views = self.net(h.cuda())
        return h_views

    def _make_view(self, v):
        dims_net = self.layer_size[v]
        net1 = nn.Sequential()
        w = torch.nn.Linear(self.lsd_dim, dims_net[0])
        a = torch.nn.ReLU()
        nn.init.xavier_normal_(w.weight)
        nn.init.constant_(w.bias, 0.0)
        net1.add_module('lin' + str(0), w)
        net1.add_module('act' + str(0), a)
        for num in range(1, len(dims_net)):
            w = torch.nn.Linear(dims_net[num - 1], dims_net[num])
            nn.init.xavier_normal_(w.weight)
            nn.init.constant_(w.bias, 0.0)
            net1.add_module('lin' + str(num), w)
            net1.add_module('act' + str(num), a)
            net1.add_module('drop' + str(num), torch.nn.Dropout(p=0.1))

        return net1


class DialogueRNN_backup(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNN_backup, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e, context_attention, D_a, dropout)

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type())  # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_p).type(U.type())  # batch, party, D_p
        e_ = torch.zeros(0).type(U.type())  # batch, D_e
        e = e_
        c_ = torch.zeros(0).type(U.type()).cuda()
        c = c_

        alpha = []
        for u_, qmask_ in zip(U, qmask):
            g_, q_, e_, c_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)], 0)
            e = torch.cat((e, e_.unsqueeze(0)), dim=0)
            c = torch.cat((c, c_.unsqueeze(0)), dim=0)
            if type(alpha_) != type(None):
                alpha.append(alpha_[:, 0, :])

        return e, c, alpha  # seq_len, batch, D_e

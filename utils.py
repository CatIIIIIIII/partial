import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
import torch


def get_sn(view_num, len_data, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param len_data:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return:Sn
    """
    matrix = None
    one_rate = 1 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(len_data, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(len_data, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(len_data, 1))).toarray()
        one_num = view_num * len_data * one_rate - len_data
        ratio = one_num / (view_num * len_data)
        matrix_iter = (randint(0, 100, size=(len_data, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * len_data)
        matrix_iter = (randint(0, 100, size=(len_data, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * len_data)
        error = abs(one_rate - ratio)
    return matrix


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    a = np.random.uniform(low, high, (fan_in, fan_out))
    a = a.astype('float32')
    a = torch.from_numpy(a).cuda()
    return a


def ave(lsd1, lsd2, label_1hot):
    """In most cases, this method is used to predict the highest accuracy.
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label_1hot: label of train set
    :return: Predicted label
    """
    F_h_h = torch.mm(lsd2, lsd1.T)
    # should sub 1.Avoid numerical errors; the number of samples of per label
    label_num = label_1hot.sum(0, keepdim=True)
    label_1hot = label_1hot.float()
    F_h_h_sum = torch.mm(F_h_h, label_1hot)
    F_h_h_mean = F_h_h_sum / label_num
    gt1 = torch.max(F_h_h_mean, dim=1)[1]
    gt_ = gt1.type(torch.IntTensor)
    gt_ = gt_.cuda()
    gt_ = gt_.reshape([gt_.shape[0], 1])

    return gt_.cpu().detach().numpy()

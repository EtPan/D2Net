import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from functools import partial


class Bandwise(object): 
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
            y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)    
    return np.mean(np.real(np.arccos(tmp)))

def cal_ergas(x_true, x_pred):
    """
    Calculate ERGAS, ERGAS offers a global indication of the quality of fused image.The ideal value is 0.
    :param x_true:
    :param x_pred:
    :return:
    """
    sum_ergas = 0
    C = x_true.shape[-3]
    for ch in range(C):
        vec_x = torch.squeeze(x_true[...,ch,:,:].data).cpu().numpy()
        vec_y = torch.squeeze(x_pred[...,ch,:,:].data).cpu().numpy()
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        tmp = r_mse / (np.mean(vec_x)**2)
        sum_ergas += tmp
    return np.sqrt(sum_ergas / x_true.shape[0])

cal_bwssim = Bandwise(compare_ssim)
cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=1))

def MSIQA(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    ergas=cal_ergas(X,Y)
    return psnr, ssim, sam, ergas

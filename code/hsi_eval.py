import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from utility import *
from hsi_setup import Engine, train_options
import models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


prefix = 'test'

if __name__ == '__main__':
    """Training settings"""
    opt = train_options()
    print(opt)

    cuda = not opt.no_cuda
    opt.no_log = True 

    basefolder = './Data/'
    resfolder = "./Result/"
    valid1 = ['icvl_512_70','icvl_512_blind']
    valid2 = ['icvl_512_deadline']
    
    for i in range(len(valid1)):
        opt.resumePath = './checkpoints/model_ckpt1.pth'
        engine = Engine(opt)
        datadir = os.path.join(basefolder, valid1[i])
        resdir = os.path.join(resfolder, valid1[i])
        if not os.path.exists(resdir):
            os.mkdir(resdir)
        mat_dataset = MatDataFromFolder(datadir, size=None)
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt', ),])
        mat_dataset = TransformDataset(mat_dataset, mat_transform)
        mat_loader = DataLoader(mat_dataset,batch_size=1, shuffle=False,num_workers=1, pin_memory=cuda)    
        res_arr, input_arr = engine.test_develop(mat_loader, savedir=resdir, verbose=True)
        print(valid1[i], res_arr.mean(axis=0))

    for i in range(len(valid2)):
        opt.resumePath = './checkpoints/model_ckpt2.pth'
        engine = Engine(opt)
        datadir = os.path.join(basefolder, valid2[i])
        resdir = os.path.join(resfolder, valid2[i])
        if not os.path.exists(resdir):
            os.mkdir(resdir)
        mat_dataset = MatDataFromFolder(datadir, size=None)
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt', ),])
        mat_dataset = TransformDataset(mat_dataset, mat_transform)
        mat_loader = DataLoader(mat_dataset,batch_size=1, shuffle=False,num_workers=1, pin_memory=cuda)    
        res_arr, input_arr = engine.test_develop(mat_loader, savedir=resdir, verbose=True)
        print(valid2[i], res_arr.mean(axis=0))
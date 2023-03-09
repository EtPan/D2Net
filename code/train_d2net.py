import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from utility import *
from hsi_setup import Engine, train_options, make_dataset


if __name__ == '__main__':
    """Training settings"""
    
    opt = train_options()
    print(opt)

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    common_transform_1 = lambda x: x
    common_transform_2 = Compose([partial(rand_crop, cropx=32, cropy=32),])

    target_transform = HSI2Tensor()
    
    sigmas = [30, 50, 70]   
    train_transform_1 = Compose([AddNoise(50),HSI2Tensor()])
    train_transform_2 = Compose([AddNoiseBlind(sigmas), HSI2Tensor()])
    train_transform_3 = Compose([AddNoiseNoniid2(), HSI2Tensor()])
    train_transform_4 = Compose([SequentialSelect(transforms=[AddNoiseImpulse(),
        AddNoiseStripe(),AddNoiseDeadline(),AddNoiseComplex()]),HSI2Tensor()])

    print('==> Preparing data..')
    icvl_64_31_TL_1 = make_dataset(opt, train_transform_1,
        target_transform, common_transform_1, opt.batchSize)

    icvl_64_31_TL_2 = make_dataset(opt, train_transform_2,
        target_transform, common_transform_1, opt.batchSize)

    icvl_64_31_TL_3 = make_dataset(opt, train_transform_3,
        target_transform, common_transform_1, opt.batchSize)

    icvl_64_31_TL_4 = make_dataset(opt, train_transform_4,
        target_transform, common_transform_2, opt.batchSize*4)
    
    
    """Test-Dev"""
    basefolder = './Data/'
    mat_names = ['icvl_512_70', 'icvl_512_blind','icvl_512_noniid_pet', 'icvl_512_deadline_pet']

    mat_datasets = [MatDataFromFolder(os.path.join(
        basefolder, name), size=5) for name in mat_names]

    if not engine.get_net().use_2dconv:
        mat_transform = Compose([LoadMatHSI(input_key='input', gt_key='gt',
                    transform=lambda x:x[:, ...][None]),])
    else:
        mat_transform = Compose([LoadMatHSI(input_key='input', gt_key='gt'),])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]
    
    mat_loaders = [DataLoader(mat_dataset,batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda) for mat_dataset in mat_datasets]
    

    """Main loop"""
    adjust_learning_rate(engine.optimizer, opt.lr)    
    epoch_per_save = 25    
    print(mat_names[0])    
    while engine.epoch < 100:
        np.random.seed() # reset seed per epoch, otherwise the noise will be added with a specific pattern
        gaps = [i for i in range(0,100) if i%25==0]
        for g in range(len(gaps)):
            if engine.epoch == gaps[g]:
                adjust_learning_rate(engine.optimizer, opt.lr)        
            if engine.epoch == gaps[g]+15:
                adjust_learning_rate(engine.optimizer, opt.lr*0.1)        
            if engine.epoch == gaps[g]+20:
                adjust_learning_rate(engine.optimizer, opt.lr*0.01)
                
        if engine.epoch <= 25:
            engine.train(icvl_64_31_TL_1)
            engine.validate(mat_loaders[0], mat_names[0])
        elif engine.epoch <= 50:
            engine.train(icvl_64_31_TL_2)
            engine.validate(mat_loaders[0], mat_names[0])
            engine.validate(mat_loaders[1], mat_names[1])
        elif engine.epoch <= 75:
            engine.train(icvl_64_31_TL_3)
            engine.validate(mat_loaders[2], mat_names[2])
        else:
            engine.train(icvl_64_31_TL_4)
            engine.validate(mat_loaders[2], mat_names[2])
            engine.validate(mat_loaders[3], mat_names[3])
        
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(model_out_path=model_latest_path)

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()
import torch
import torch.optim as optim
import models
import os
import argparse
from os.path import join
from utility import *
from scipy.io import savemat
from os.path import basename, exists


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def train_options():
    parser = argparse.ArgumentParser()
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args    
    parser.add_argument('--prefix', '-p', type=str, default='denoise',
                        help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names))
    parser.add_argument('--batchSize', '-b', type=int,
                        default=16, help='training batch size. default=16')         
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate. default=1e-3.')
    parser.add_argument('--loss', type=str, default='l1',
                        help='which loss to choose.', choices=['l1', 'l2'])
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true',
                        help='disable logger?')
    parser.add_argument('--seed', type=int, default=2022,
                        help='random seed to use.')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')                                     
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--dataroot', '-d', type=str,
                        default='./ICVL64_31.db', help='data root')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                            help='not resume optimizer')          
    parser.add_argument('--chop', action='store_true',
                            help='forward chop') 
    parser.add_argument('--gpu-ids', type=str, default='1', help='gpu ids')
    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    return opt


def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    dataset = TransformDataset(dataset, common_transform)
    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=8, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)

    return train_loader


class Engine(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoints', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0
        self.iteration = 0

        cuda = not self.opt.no_cuda
        self.device = 'cuda' if cuda else 'cpu'
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        self.net = models.__dict__[self.opt.arch]()
        init_params(self.net, init_type='kn') 
        
        if self.opt.loss == 'l2':
            self.criterion = nn.MSELoss()
        if self.opt.loss == 'l1':
            self.criterion = nn.L1Loss()
        
        print("Loss:",self.criterion)

        if cuda:
            self.net.to(self.device)
            self.criterion = self.criterion.to(self.device)

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs'), self.opt.prefix)

        """Optimization Setup"""
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.opt.lr, amsgrad=False)

        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath, not self.opt.no_ropt)
        else:
            print('==> Building model..')
            #print(self.net)

    def forward(self, inputs):        
        if self.opt.chop:           
            output = self.forward_chop(inputs)
        else:
            output = self.net(inputs)
        
        return output

    def forward_chop(self, x, base=16):        
        #n, c, b, h, w = x.size()
        if self.net.use_2dconv:
            n, b, h, w = x.size()
        else:
            n, c, b, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)        
        
        inputs = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.net(input_i) for input_i in inputs]

        output = torch.zeros_like(x)
        output_w = torch.zeros_like(x)
        
        output[..., 0:h_half, 0:w_half] += outputs[0][..., 0:h_half, 0:w_half]
        output_w[..., 0:h_half, 0:w_half] += 1
        output[..., 0:h_half, w_half:w] += outputs[1][..., 0:h_half, (w_size - w + w_half):w_size]
        output_w[..., 0:h_half, w_half:w] += 1
        output[..., h_half:h, 0:w_half] += outputs[2][..., (h_size - h + h_half):h_size, 0:w_half]
        output_w[..., h_half:h, 0:w_half] += 1
        output[..., h_half:h, w_half:w] += outputs[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        output_w[..., h_half:h, w_half:w] += 1
        
        output /= output_w

        return output

    def __step(self, train, inputs, targets):        
        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        total_norm = None
        if self.get_net().bandwise:
            O = []
            for time, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                o = self.net(i)
                O.append(o)
                loss = self.criterion(o, t)
                if train:
                    loss.backward()
                loss_data += loss.item()
            outputs = torch.cat(O, dim=1)
        else:
            outputs = self.net(inputs) 
            
            loss = self.criterion(outputs, targets)
            
            if train:
                loss.backward()
            loss_data += loss.item()
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), 1e6)
            self.optimizer.step()

        return outputs, loss_data, total_norm

    def load(self, resumePath=None, load_opt=True):
        model_best_path = join(self.basedir, self.prefix, 'model_latest.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path)
        #### comment when using memnet
        self.epoch = checkpoint['epoch'] 
        self.iteration = checkpoint['iteration']
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        ####
        self.get_net().load_state_dict(checkpoint['net'])

    def train(self, train_loader):
        print('\nEpoch: %d' % self.epoch)
        self.net.train()
        train_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.to(self.device), targets.to(self.device)            
            outputs, loss_data, total_norm = self.__step(True, inputs, targets)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx+1)

            if not self.opt.no_log:
                self.writer.add_scalar(
                    join(self.prefix, 'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(
                    join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)

            self.iteration += 1

            progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e' 
                         % (avg_loss, loss_data, total_norm))

        self.epoch += 1
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)

    def validate(self, valid_loader, name):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        print('[i] Eval dataset {}...'.format(name))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)                

                outputs, loss_data, _ = self.__step(False, inputs, targets)
                psnr = np.mean(cal_bwpsnr(outputs, targets))

                validate_loss += loss_data
                avg_loss = validate_loss / (batch_idx+1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx+1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f'
                            % (avg_loss, avg_psnr))

        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss

    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iteration))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
        }
        
        state.update(kwargs)

        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    # saving result into disk
    def test_develop(self, test_loader, savedir=None, verbose=True):

        def torch2numpy(hsi):
            if self.net.use_2dconv:
                R_hsi = hsi.data[0].cpu().numpy().transpose((1,2,0))
            else:
                R_hsi = hsi.data[0].cpu().numpy()[0,...].transpose((1,2,0))
            return R_hsi    

        self.net.eval()
        test_loss = 0
        dataset = test_loader.dataset.dataset

        res_arr = np.zeros((len(test_loader), 4+1))
        input_arr = np.zeros((len(test_loader), 4+1))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()              
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs, loss_data, _ = self.__step(False, inputs, targets)
                
                torch.cuda.synchronize()
                end.record()
                torch.cuda.synchronize()
                test_loss += loss_data
                
                res_arr[batch_idx, :4] = MSIQA(outputs, targets)
                input_arr[batch_idx, :4] = MSIQA(inputs, targets)
                res_arr[batch_idx, 4:] = start.elapsed_time(end)

                if verbose:
                    print("================",batch_idx,"psnr,ssim,sam,ergas,time")
                    print("---noisy:", input_arr[batch_idx, 0], input_arr[batch_idx, 1],input_arr[batch_idx, 2],input_arr[batch_idx, 3])
                    print("denoised:", res_arr[batch_idx, 0], res_arr[batch_idx, 1],res_arr[batch_idx, 2],res_arr[batch_idx, 3],res_arr[batch_idx, 4])

                if savedir:
                    filedir = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0])  
                    outpath = join(filedir, '{}.mat'.format(self.opt.arch))

                    if not exists(filedir):
                        os.mkdir(filedir)

                    if not exists(outpath):
                        savemat(outpath, {'R_hsi': torch2numpy(outputs)})
                        
        return res_arr, input_arr

    def get_net(self):
        return self.net           

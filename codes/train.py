import os
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import sys
import argparse
import re
import shutil

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss, L1Loss
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from dataloader.LeftAtrium import LAHeart
from dataloader.pancreas import Pancreas

import utils.loss
from utils import statistic, ramps
from utils.loss import DiceLoss, SoftIoULoss, to_one_hot
from utils.losses import FocalLoss
from utils.logger import get_cur_time,checkpoint_save
from utils.Generate_Prototype import *
from utils.train_util import *
from utils.SDF import compute_sdf


from networks.vnet_AMC import VNet_AMC
from networks.vnet_AMC_sdf import VNet as VNet_sdf
from networks.vnet_AMC_img import VNet as VNet_img
from test import test_LA_Pancreas, test_AD



def get_arguments():

    parser = argparse.ArgumentParser(description='Knowledge-driven Pyramid Prototype Learning for Semi-Supervised Medical Image Segmentation')

    # Model
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--exp', type=str, default='LA', help='experiment_name')
    parser.add_argument('--alpha', type=float, default=0.99, help='params in ema update')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default='../results', help='Paths to previous checkpoints')

    # dataset
    parser.add_argument("--data_dir", type=str, default='../../../Datasets/LA_dataset',
                        help="Path to the dataset.")
    parser.add_argument("--list_dir", type=str, default='../datalist/LA',
                        help="Paths to cross-validated datasets, list of test sets and all training sets (including all labeled and unlabeled samples)")
    parser.add_argument("--save_path", type=str, default='../results',
                        help="Path to save.")
    parser.add_argument("--aug_times", type=int, default=5,
                        help="times of augmentation for training.")

    # Optimization options
    parser.add_argument('--lab_batch_size', type=int,  default=1, help='batch size')
    parser.add_argument('--unlab_batch_size', type=int,  default=2, help='batch size')
    parser.add_argument('--lr', type=float,  default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int,  default=400, help='maximum epoch number to pretraining')
    parser.add_argument('--save_step', type=int,  default=5, help='frequecy of checkpoint save in pretraining')
    parser.add_argument('--consistency_rampup', type=float,
                        default=300.0, help='consistency_rampup')

    parser.add_argument('--beta1', type=float,  default=0.5, help='params of optimizer Adam')
    parser.add_argument('--beta2', type=float,  default=0.999, help='params of optimizer Adam')
    parser.add_argument('--scaler', type=float,  default=1, help='multiplier of prototype')
    
    # Miscs
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
    parser.add_argument('--seed', type=int, default=1337, help='set the seed of random initialization')
    # parser.add_argument('--seed', type=int, default=3407, help='set the seed of random initialization')

    
    return parser.parse_args()

args = get_arguments()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# create logger
resultdir = os.path.join(args.save_path, args.exp)
logdir = os.path.join(resultdir, 'logs')
savedir = os.path.join(resultdir, 'checkpoints')
shotdir = os.path.join(resultdir, 'snapshot')
print('Result path: {}\nLogs path: {}\nCheckpoints path: {}\nSnapshot path: {}'.format(resultdir, logdir, savedir, shotdir))

os.makedirs(logdir, exist_ok=True)
os.makedirs(savedir, exist_ok=True)
os.makedirs(shotdir, exist_ok=True)

writer = SummaryWriter(logdir)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

fh = logging.FileHandler(shotdir+'/'+'snapshot.log', encoding='utf8')
fh.setFormatter(formatter) 
logger.addHandler(fh)
logging.info(str(args))

def create_model(ema=False):
    net = nn.DataParallel(VNet_AMC(n_channels=1, n_classes=args.num_classes, n_branches=4))
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def create_dataloader():
    if 'LA' in args.data_dir:
        train_labset = LAHeart(args.data_dir,args.list_dir,split='lab', aug_times = args.aug_times)
        train_unlabset = LAHeart(args.data_dir,args.list_dir,split='unlab', aug_times = args.aug_times)
        testset = LAHeart(args.data_dir,args.list_dir,split='test')
    else:
        train_labset =Pancreas(args.data_dir, args.list_dir,split='lab', aug_times = args.aug_times)
        train_unlabset = Pancreas(args.data_dir,args.list_dir,split='unlab', aug_times = args.aug_times)
        testset = Pancreas(args.data_dir,args.list_dir,split='test', aug_times = args.aug_times) 
    
    trainlab_loader = DataLoader(train_labset, batch_size=args.lab_batch_size, shuffle=True, num_workers=0)
    trainunlab_loader = DataLoader(train_unlabset, batch_size=args.unlab_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    logging.info("{} iterations for lab per epoch.".format(len(trainlab_loader)))
    logging.info("{} iterations for unlab per epoch.".format(len(trainunlab_loader)))
    logging.info("{} samples for test.\n".format(len(test_loader)))
    return trainlab_loader, trainunlab_loader, test_loader


def main():
    save_path = Path(savedir)

    set_random_seed(args.seed)
    # set_define_seed(args.seed)
    sdf_net = VNet_sdf(n_channels=1, n_classes=args.num_classes - 1, normalization='instancenorm', has_dropout=True,has_skipconnect=False).cuda()
    img_net = VNet_img(n_channels=1, n_classes=1, normalization='instancenorm', has_dropout=True,has_skipconnect=False).cuda()
    net = create_model().cuda()
    ema_net = create_model(ema=True).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)) 
    optimizer2 = optim.Adam(sdf_net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)) 
    optimizer3 = optim.Adam(img_net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    if args.resume:
        load_net_opt(net, optimizer, Path(args.load_path) / 'best.pth')
        load_net_opt(ema_net, optimizer, Path(args.load_path) / 'best_ema.pth')

    trainlab_loader, trainunlab_loader, test_loader = create_dataloader()

    dice_loss = DiceLoss(nclass=args.num_classes)
    ce_loss = CrossEntropyLoss()
    focal_loss = FocalLoss()
    mse_loss = MSELoss()
    
    l1_loss = L1Loss()
    iou_loss = SoftIoULoss(nclass=args.num_classes)
    pixel_level_ce_loss = CrossEntropyLoss(reduction='none')
    pixel_level_mse_loss = MSELoss(reduction='none')


    maxdice = 0
    iter_num = 0

    for epoch in tqdm(range(args.num_epochs), ncols=70):
        logging.info('\n')

        net.train()
        for step, (labeled_batch, unlabeled_batch) in enumerate(zip(trainlab_loader,trainunlab_loader)):
            lab_img, lab_lab, lab_img_norm = labeled_batch['image'].cuda(), labeled_batch['label'].cuda(), labeled_batch['image_norm'].cuda()
            unlab_img = unlabeled_batch['image'].cuda()
            lab_lab_onehot = to_one_hot(lab_lab.unsqueeze(1), args.num_classes)
            '''sdf part'''
            lab_sdf_out = sdf_net(lab_img)
            lab_sdf = lab_sdf_out[0]
            gt_sdm = compute_sdf(lab_lab[:].cpu().numpy(), lab_sdf[:, 0, ...].shape)
            gt_sdm = torch.from_numpy(gt_sdm).float().cuda()
            loss_sdfl2 = mse_loss(lab_sdf[:, 0, ...], gt_sdm)
            lab_fts_sdf_1 = F.interpolate(lab_sdf_out[1], size=lab_lab.shape[-3:],mode='trilinear')
            lab_prototypes_sdf_1 = getPrototype(lab_fts_sdf_1, lab_lab_onehot)
            lab_fts_sdf_2 = F.interpolate(lab_sdf_out[2], size=lab_lab.shape[-3:],mode='trilinear')
            lab_prototypes_sdf_2 = getPrototype(lab_fts_sdf_2, lab_lab_onehot)
            lab_fts_sdf_3 = F.interpolate(lab_sdf_out[3], size=lab_lab.shape[-3:],mode='trilinear')
            lab_prototypes_sdf_3 = getPrototype(lab_fts_sdf_3, lab_lab_onehot)

            '''img part'''
            lab_img_out = img_net(lab_img)
            lab_img_out_ce = lab_img_out[0]
            # loss_imgl2 = mse_loss(lab_img_out_ce, lab_img_norm)
            loss_imgl2 = mse_loss(lab_img_out_ce, lab_img_norm.unsqueeze(1))

            lab_fts_img_1 = F.interpolate(lab_img_out[1], size=lab_lab.shape[-3:],mode='trilinear')
            lab_prototypes_img_1 = getPrototype(lab_fts_img_1, lab_lab_onehot)
            lab_fts_img_2 = F.interpolate(lab_img_out[2], size=lab_lab.shape[-3:],mode='trilinear')
            lab_prototypes_img_2 = getPrototype(lab_fts_img_2, lab_lab_onehot)
            lab_fts_img_3 = F.interpolate(lab_img_out[3], size=lab_lab.shape[-3:],mode='trilinear')
            lab_prototypes_img_3 = getPrototype(lab_fts_img_3, lab_lab_onehot)

            '''Supervised'''
            lab_out = net(lab_img)

            lab_out_ce = lab_out[0]
            lab_out_dice = lab_out[1]
            lab_out_focal = lab_out[2]
            lab_out_iou = lab_out[3]

            loss_ce = ce_loss(lab_out_ce, lab_lab)
            loss_dice = dice_loss(lab_out_dice, lab_lab)
            loss_focal = focal_loss(lab_out_focal, lab_lab)
            # loss_iou = ce_loss(lab_out_iou, lab_lab)
            loss_iou = iou_loss(lab_out_iou, lab_lab)

            loss_supervised = (loss_ce + loss_dice + loss_focal + loss_iou) / 4 

            # labeled prototypes
            lab_fts_1 = F.interpolate(lab_out[4], size=lab_lab.shape[-3:],mode='trilinear')
            lab_prototypes_1 = getPrototype(lab_fts_1, lab_lab_onehot)
            lab_fts_2 = F.interpolate(lab_out[5], size=lab_lab.shape[-3:],mode='trilinear')
            lab_prototypes_2 = getPrototype(lab_fts_2, lab_lab_onehot)
            lab_fts_3 = F.interpolate(lab_out[6], size=lab_lab.shape[-3:],mode='trilinear')
            lab_prototypes_3 = getPrototype(lab_fts_3, lab_lab_onehot)

            '''Unsupervised'''
            with torch.no_grad():
                unlab_ema_out = ema_net(unlab_img)
                unlab_ema_out_pred = (unlab_ema_out[0] + unlab_ema_out[1] + unlab_ema_out[2] + unlab_ema_out[3]) / 4

                unlab_ema_out_soft = torch.softmax(unlab_ema_out_pred, dim=1)
                unlab_ema_mask = torch.argmax(unlab_ema_out_soft, dim=1)
                unlab_ema_mask_onehot = to_one_hot(unlab_ema_mask.unsqueeze(1), args.num_classes)

                # uncertainty assesment
                uncertainty =  -torch.sum(unlab_ema_out_soft * torch.log(unlab_ema_out_soft  + 1e-16), dim=1)
                norm_uncertainty = torch.stack([uncertain / torch.sum(uncertain) for uncertain in uncertainty],dim=0)

                reliability_map = (1 - norm_uncertainty) / np.prod(np.array(norm_uncertainty.shape[-3:]))

                unlab_ema_fts_1 = F.interpolate(unlab_ema_out[4], size=unlab_ema_mask.shape[-3:],mode='trilinear')
                unlab_prototypes_1 = getPrototype(unlab_ema_fts_1, unlab_ema_mask_onehot, reliability_map)
                unlab_ema_fts_2 = F.interpolate(unlab_ema_out[5], size=unlab_ema_mask.shape[-3:],mode='trilinear')
                unlab_prototypes_2 = getPrototype(unlab_ema_fts_2, unlab_ema_mask_onehot, reliability_map)
                unlab_ema_fts_3 = F.interpolate(unlab_ema_out[6], size=unlab_ema_mask.shape[-3:],mode='trilinear')
                unlab_prototypes_3 = getPrototype(unlab_ema_fts_3, unlab_ema_mask_onehot, reliability_map)

                consistency_weight = get_current_consistency_weight(epoch, args.consistency_rampup)

            '''Prototype fusion'''
            lab_prototypes_aux_1 = [(lab_prototypes_sdf_1[c] +lab_prototypes_img_1[c])/2 for c in range(args.num_classes)]
            lab_prototypes_aux_2 = [(lab_prototypes_sdf_2[c] +lab_prototypes_img_2[c])/2 for c in range(args.num_classes)]
            lab_prototypes_aux_3 = [(lab_prototypes_sdf_3[c] +lab_prototypes_img_3[c])/2 for c in range(args.num_classes)]


            prototypes_1 = [ ((lab_prototypes_1[c] + lab_prototypes_aux_1[c] + consistency_weight * unlab_prototypes_1[c]) / (2 + consistency_weight)) for c in range(args.num_classes)]
            prototypes_2 = [ ((lab_prototypes_2[c] + lab_prototypes_aux_2[c] + consistency_weight * unlab_prototypes_2[c]) / (2 + consistency_weight)) for c in range(args.num_classes)]
            prototypes_3 = [ ((lab_prototypes_3[c] + lab_prototypes_aux_3[c] + consistency_weight * unlab_prototypes_3[c]) / (2 + consistency_weight)) for c in range(args.num_classes)]



            lab_dist_1 =  torch.stack([calDist(lab_fts_1, prototype, scaler=args.scaler) for prototype in prototypes_1], dim=1)
            lab_dist_2 =  torch.stack([calDist(lab_fts_2, prototype, scaler=args.scaler) for prototype in prototypes_2], dim=1)
            lab_dist_3 =  torch.stack([calDist(lab_fts_3, prototype, scaler=args.scaler) for prototype in prototypes_3], dim=1)
            lab_dist = torch.mean(torch.concat([lab_dist_1.unsqueeze(1), lab_dist_2.unsqueeze(1), lab_dist_3.unsqueeze(1)], dim=1), dim=1)

            unlab_dist_1 = torch.stack([calDist(unlab_ema_fts_1, prototype, scaler=args.scaler) for prototype in prototypes_1], dim=1)
            unlab_dist_2 = torch.stack([calDist(unlab_ema_fts_2, prototype, scaler=args.scaler) for prototype in prototypes_2], dim=1)
            unlab_dist_3 = torch.stack([calDist(unlab_ema_fts_3, prototype, scaler=args.scaler) for prototype in prototypes_3], dim=1)
            unlab_dist = torch.mean(torch.concat([unlab_dist_1.unsqueeze(1), unlab_dist_2.unsqueeze(1), unlab_dist_3.unsqueeze(1)], dim=1), dim=1)
                
            '''Prototype consistency learning'''
            loss_pc_lab = ce_loss(lab_dist, lab_lab)
            loss_pc_lab_1 = ce_loss(lab_dist_1, lab_lab)
            loss_pc_lab_2 = ce_loss(lab_dist_2, lab_lab)
            loss_pc_lab_3 = ce_loss(lab_dist_3, lab_lab)

            loss_pc_lab_all = (loss_pc_lab + loss_pc_lab_1 + loss_pc_lab_2 + loss_pc_lab_3) / 4

            loss_pc_unlab = torch.sum(pixel_level_ce_loss(unlab_dist, unlab_ema_mask) * reliability_map)
            loss_pc_unlab_1 = torch.sum(pixel_level_mse_loss(unlab_dist_1, unlab_dist) * (reliability_map.unsqueeze(1)))
            loss_pc_unlab_2 = torch.sum(pixel_level_mse_loss(unlab_dist_2, unlab_dist) * (reliability_map.unsqueeze(1)))
            loss_pc_unlab_3 = torch.sum(pixel_level_mse_loss(unlab_dist_3, unlab_dist) * (reliability_map.unsqueeze(1)))

            loss_pc_unlab_all = (loss_pc_unlab + loss_pc_unlab_1 + loss_pc_unlab_2 + loss_pc_unlab_3) / 4

            '''Total'''
            # loss = loss_supervised + loss_pc_lab + consistency_weight * loss_pc_unlab + loss_sdf
            loss = loss_supervised + loss_pc_lab_all + consistency_weight * loss_pc_unlab_all + (loss_sdfl2 +loss_imgl2)/2


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()
            optimizer3.step()
            update_ema_variables(net, ema_net, args.alpha, iter_num)
            iter_num += 1

            lab_masks = torch.softmax(lab_out_dice, dim = 1)
            lab_masks = torch.argmax(lab_masks, dim = 1)
            train_dice = statistic.dice_ratio(lab_masks, lab_lab)

            logging.info('epoch : %d, step : %d, loss_all: %.4f,'
                         'loss_ce: %.4f, loss_dice: %.4f, loss_focal: %.4f, loss_iou: %.4f, '
                         'loss_supervised: %.4f, loss_pc_lab: %.4f, '
                         'loss_pc_unlab: %.4f, train_dice: %.4f, loss_sdf: %.4f,  loss_img: %.4f' % (
                            epoch+1, step, loss.item(), 
                            loss_ce.item(), loss_dice.item(), loss_focal.item(), loss_iou.item(),
                            loss_supervised.item(), loss_pc_lab.item(), loss_pc_unlab.item(), train_dice, loss_sdfl2.item(), loss_imgl2.item()))           

            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_focal', loss_focal, iter_num)
            writer.add_scalar('info/loss_iou', loss_iou, iter_num)
            writer.add_scalar('info/loss_supervised', loss_supervised, iter_num)

            writer.add_scalar('info/loss_sdf', loss_sdfl2, iter_num)
            # writer.add_scalar('info/loss_sdfseg', loss_sdfseg, iter_num)
            writer.add_scalar('info/loss_img', loss_imgl2, iter_num)



            writer.add_scalar('info/loss_pc_lab', loss_pc_lab, iter_num)
            writer.add_scalar('info/loss_pc_unlab', loss_pc_unlab, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            writer.add_scalar('info/loss_all', loss, iter_num)
            writer.add_scalar('val/train_dice', train_dice, iter_num)
            

        '''Test'''
        if (epoch+1) % args.save_step == 0:
            if 'TBAD' in args.data_dir:
                val_dice, maxdice, max_flag = test_AD(net, test_loader, args, maxdice)
            else:
                val_dice, maxdice, max_flag = test_LA_Pancreas(net, test_loader,args, maxdice)

            writer.add_scalar('val/test_dice', val_dice, epoch+1)

            save_mode_path = os.path.join(save_path,
                            'epoch_{}_iter_{}_dice_{}.pth'.format(
                                epoch+1, iter_num, round(val_dice, 4)))
            save_net_opt(net, optimizer, save_mode_path, epoch+1)

            save_ema_path = os.path.join(save_path,
                            'ema_epoch_{}_iter_{}_dice_{}.pth'.format(
                                epoch+1, iter_num, round(val_dice, 4)))
            save_net_opt(ema_net, optimizer, save_ema_path, epoch+1)


            if max_flag:
                save_net_opt(net, optimizer, save_path / 'best.pth', epoch+1)
                save_net_opt(ema_net, optimizer, save_path / 'best_ema.pth', epoch+1)

        writer.flush()



if __name__ == '__main__':
    if os.path.exists(resultdir + '/codes'):
        shutil.rmtree(resultdir + '/codes')
    shutil.copytree('.', resultdir + '/codes',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    main()

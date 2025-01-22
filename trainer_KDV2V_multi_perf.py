import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import wandb
from tqdm import tqdm
from logger.logger import setup_logging, get_logger
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from torchmetrics import Accuracy
from utils.util_helper import get_gaussian_kernel
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter
from losses import BCELoss


class Trainer:
    def __init__(self,
                 cfg, num_tasks, task_id, model, match_optimizer, loc_optimizer, scheduler,
                 train_loader, test_loader_list,
                 acc_matrix, niter,
                 arg_config=None,
                 device=None, device_ids=[]):
        # pass
        
        self.niter = niter
        self.num_tasks = num_tasks
        self.task_id = task_id
        self.acc_matrix = acc_matrix

        self.train_cfg = cfg.trainer
        self.saver_cfg = cfg.saver

        self.epochs = self.train_cfg.epochs
        self.match_epochs = self.train_cfg.match_epochs
        self.log_step = self.train_cfg.log_step
        self.device = device
        self.device_ids = device_ids
        self.val_freq_epoch = self.train_cfg.val_freq_epoch
        
        self.model = model
        self.match_optimizer = match_optimizer
        self.loc_optimizer = loc_optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.test_loader_list = test_loader_list

        self.ce_loss = nn.CrossEntropyLoss()
        self.seg_criteria = BCELoss()
    
        self.board_dir = self.saver_cfg.board_dir
        self.model_dir = self.saver_cfg.model_dir
        self.visual_dir = self.saver_cfg.visual_dir

        self.arg_config = arg_config
        self.distributed = arg_config.distributed
        if self.distributed:
            self.local_master = (arg_config.local_rank == 0)
            self.global_master = (dist.get_rank() == 0)
        else:
            self.local_master = True
            self.global_master = True

        if self.local_master:
            if not os.path.exists(self.board_dir):
                os.makedirs(self.board_dir, exist_ok=True)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir, exist_ok=True)
            if not os.path.exists(self.visual_dir):
                os.makedirs(self.visual_dir, exist_ok=True)
        
            self.logger = get_logger('trainer')

            self.writer = SummaryWriter(log_dir=self.board_dir)

    def logger_info(self, msg):
        self.logger.info(msg) if self.local_master else None

    def logger_warning(self, msg):
        self.logger.warning(msg) if self.local_master else None

    def ortho_penalty(self, t):
        return ((t @t.T - torch.eye(t.shape[0]).cuda())**2)
    

    def wbshow(self, ShowImg):

        if ShowImg.dim() == 4 and ShowImg.shape[1] > 3: # feature
            ShowImg = torch.mean(ShowImg, dim=1)
        
        if ShowImg.dim() == 4 and ShowImg.shape[1] == 3: # Image
            ShowImg = ShowImg.permute(0,2,3,1)

        ShowImg = (ShowImg - ShowImg.min()) / (ShowImg.max() - ShowImg.min())
        ShowImgs = torch.chunk(ShowImg, chunks=ShowImg.shape[0], dim=0)
        show_sub_imgs = [wandb.Image(t.squeeze(0).squeeze(0).detach().cpu().numpy()) for t in ShowImgs]

        return show_sub_imgs


    def train(self, task_id):
        best_pmap, best_iauc, best_pauc = 0.0, 0.0, 0.0
        self.logger_info('training begin')

        if not os.path.exists(os.path.join(self.model_dir, 'model.pth')):
            start_epoch = -1
        else:
            break_ckpt = torch.load(os.path.join(self.model_dir, 'model.pth'), map_location=self.device)
            self.model.load_state_dict(break_ckpt['net'])
            self.optim.load_state_dict(break_ckpt['optimizer'])
            self.scheduler.load_state_dict(break_ckpt['lr_schedule'])
            start_epoch = break_ckpt['epoch']

        for epoch in range(start_epoch+1, max(self.epochs[task_id], self.match_epochs[task_id])):
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)
                # self.test_loader.sampler.set_epoch(epoch)

            self.train_epoch(epoch, task_id)
            

            if epoch == self.epochs[task_id] - 1:
                self.resume_ckpt('_{}.pth'.format(epoch), task_id, epoch)
                
            
    def train_epoch(self, epoch, task_id):
        self.model.train()
        
        # if self.distributed:
        #     self.model.module.Image_clip.eval()

        #     for _, param in self.model.module.Image_clip.named_parameters():
        #         param.requires_grad = False

        # else:
        #     self.model.Image_clip.eval()
        #     for _, param in self.model.Image_clip.named_parameters():
        #         param.requires_grad = False
        accumulation_step = 2
        # self.match_optimizer.zero_grad()
        # self.loc_optimizer.zero_grad()
        
        for step_idx, sample in enumerate(tqdm(self.train_loader)):

            for key, input_value in sample.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    sample[key] = input_value.to(self.device)

            prompt_id, id_counts, batched_key_norm, x_embed_norm, simmap = self.model(sample['img_aug'], sample['img_origin'], True, task_id)  # img_origin
            
            Simmap = [F.interpolate(_simmap, size=sample['img_aug'].shape[2:], mode='bilinear') for _simmap in simmap]
            seg1_loss = self.seg_criteria(Simmap[0], sample['mask'], 'normal')
            seg2_loss = self.seg_criteria(Simmap[1], sample['mask'], 'normal')
            seg3_loss = self.seg_criteria(Simmap[2], sample['mask'], 'normal')

            seg_loss = seg1_loss + seg2_loss + seg3_loss
            
            cosine_loss = 1 - F.cosine_similarity(
                torch.flatten(x_embed_norm, 1).detach(),
                torch.flatten(batched_key_norm, 1), dim=1).mean(0)

            if epoch < self.match_epochs[task_id]:
                self.match_optimizer.zero_grad()
                cosine_loss.backward()
                self.match_optimizer.step()

            if epoch < self.epochs[task_id]:
                self.loc_optimizer.zero_grad()
                seg_loss.backward()
                self.loc_optimizer.step()

            self.niter += 1

            if step_idx % self.log_step == 0:
                if epoch < self.epochs[task_id]:
                    self.logger_info(
                        f"Train Epoch:[{epoch}/{max(self.epochs[task_id], self.match_epochs[task_id])}] Step:[{step_idx}/{len(self.train_loader)}] "
                        f"Similarity_loss:{cosine_loss.mean().item():.6f} Segment_loss:{seg_loss.mean().item():.6f}") # 
                else:
                    self.logger_info(
                        f"Train Epoch:[{epoch}/{max(self.epochs[task_id], self.match_epochs[task_id])}] Step:[{step_idx}/{len(self.train_loader)}] "
                        f"Similarity_loss:{cosine_loss.mean().item():.6f}") # 
                    
            if self.niter % 500 == 0 and self.local_master:
                wandb.log({
                    'train/sim_loss': cosine_loss,
                    'train/seg_loss': seg_loss,
                }, step=self.niter)


        if self.scheduler is not None:
            self.scheduler.step()
            self.logger_info(f"learning rate: {self.match_optimizer.state_dict()['param_groups'][1]['lr']}")


    def ad_eval_metric(self,gt, pred, img_gt, img_pred):
        # pixel-auc, pixel-IOU, pixel-pro
        _rocauc = roc_auc_score(gt.flatten(), pred.flatten())
        _img_rocauc = roc_auc_score(img_gt.flatten(), img_pred.flatten())        
        _ap = average_precision_score(gt.flatten(), pred.flatten())
        return _rocauc, _ap, _img_rocauc

    def resume_ckpt(self, ckpt_name, task_id, epoch):

        if self.distributed:
            dist.barrier()
        if not self.local_master:
            return

        if hasattr(self.model, 'module'):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        if self.scheduler is not None:
            checkpoint = {
                "net": state_dict,
                # 'lr_schedule':self.scheduler.state_dict(),
                "epoch": epoch
                }
        else:
            checkpoint = {
                "net": state_dict,
                "epoch": epoch
                } 
        
        torch.save(checkpoint, os.path.join(self.model_dir, 'Task{}'.format(task_id) + ckpt_name))
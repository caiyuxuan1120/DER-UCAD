import os
import sys
import argparse
from clip import tokenize
import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, Sampler
import clip
import yaml

from torch.nn.parallel import DistributedDataParallel as DDP

from easydict import EasyDict
from utils.util_helper import set_random_seed
from logger.logger import setup_logging, get_logger
from datasets.mvtec_visa_datasets import MVTecVisaTrainDataset, MVTecVisaTestDataset
from models.model_helper import ModelHelper
from trainer.trainer_KDV2V_multi_perf import Trainer
from logger.logger import get_logger
import torch.distributed as dist

from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)


MVTEC_TEXTURE_CATEGORY = ['tile', 'grid', 'wood', 'carpet', 'leather']
MVTEC_OBJECT_CATEGORY = ['bottle',  'capsule', 'cable', 'pill', 'metal_nut', 'screw', 'toothbrush', 'transistor', 'hazelnut', 'zipper']
MVTEC_CATEGORY = [ 'bottle', 'capsule', 'cable', 'wood', 'carpet', 'tile', 'grid', 'leather', 'pill', 'metal_nut', 'screw', 'toothbrush', 'transistor', 'hazelnut', 'zipper']
VISA_CATEGORY = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

def main(arg_config, local_master: bool, logger=None):
    with open(arg_config.config, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)
    
    # create logger
    os.makedirs(config.saver.log_dir, exist_ok=True)
    setup_logging(Path(config.saver.log_dir))
    logger = get_logger('train')

    device, device_ids = prepare_device(arg_config.local_rank, arg_config.local_world_size, arg_config.distributed)

    # create model
    model = ModelHelper(config, **config.model['kwargs']).to(device)
    

    obj_per_cls = 2
    txt_per_cls = 1
    cls_per_task = obj_per_cls + txt_per_cls
    obj_cls = ['pcb4', 'cashew', 'pcb2', 'chewinggum', 'fryum', 'pcb1', 'pcb3', 'capsules', 'macaroni1', 'macaroni2', 'pipe_fryum', 'candle']
    num_tasks = len(obj_cls) // cls_per_task
    

    seen_val_dataset = []
    niter=0

    acc_matrix = np.zeros((num_tasks, num_tasks))
    for Task in range(num_tasks):
        mvtec_dir_list = []
        mvtec_dir_list_test = []
        data_dir_list = []
        data_dir_list_test = []
        
        for idx in range(cls_per_task):
            mvtec_dir_list.append(config.dataset.mvtec_path + obj_cls[cls_per_task*Task+idx] + "/train/good/")
            mvtec_dir_list_test.append(config.dataset.mvtec_path + obj_cls[cls_per_task*Task+idx] + "/test/")

        
        # # training dataset
        dataset = MVTecVisaTrainDataset(mvtec_dir_list, dtd_dir=config.dataset.dtd_path, defect_ratio=1.0, resize_shape=[224,224])
        train_loader = DataLoader(
            dataset, batch_size=config.dataset.batch_size, shuffle=True, num_workers=config.dataset.num_workers, drop_last=True,)

        test_dataset = MVTecVISADRAEDataset(mvtec_dir_list_test, resize_shape=[224,224])
        test_loader = DataLoader(test_dataset, batch_size=config.dataset.batch_size, num_workers=config.dataset.num_workers)
        

        seen_val_dataset.append(test_loader)


        if Task > 0:
            ckpts = torch.load(os.path.join(config.saver.model_dir, 'Task{}_{}.pth'.format(Task-1, config.trainer.epochs[Task-1]-1)), map_location=device)
            model.load_state_dict(ckpts['net'])

        non_learnable_params, learnable_params = [], []
        key_params = []
        for name, params in model.named_parameters():
            if name.startswith('Text_clip') or name.startswith('Image_clip') or \
                    name.startswith('DINO_model') or name.startswith('FeatureExtrac'): # 
                non_learnable_params.append(params)
            elif name.startswith('learn_key'):
                key_params.append(params)
            else:
                print(name)
                learnable_params.append(params)

        match_optimizer = optim.AdamW([
            {'params': non_learnable_params, 'lr':0.0},
            {'params': key_params , **config.trainer['match_optimizer']},
        ])



        loc_optimizer = optim.AdamW([
            {'params': non_learnable_params, 'lr':0.0},
            {'params': learnable_params , **config.trainer['loc_optimizer']},
            ])
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(loc_optimizer, [150, ], gamma=0.2)

        if arg_config.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=device_ids, output_device=device_ids[0],
                                        find_unused_parameters=True)
            
        _trainer = Trainer(config, num_tasks, Task, 
                           model, match_optimizer, loc_optimizer, scheduler, # TODO
                           train_loader, seen_val_dataset,
                           acc_matrix=acc_matrix, niter=niter, 
                           arg_config=arg_config, device=device, device_ids=device_ids)
        
        logger.info('training start')
        _trainer.train(Task)

        niter = _trainer.niter
        acc_matrix = _trainer.acc_matrix
        # import pdb;pdb.set_trace()
    
    # wandb.finish()

def prepare_device(local_rank, local_world_size, distributed):
    '''
    setup GPU device if available, move model into configured device
    :param local_rank:
    :param local_world_size:
    :return:
    '''
    if distributed:
        ngpu_per_process = torch.cuda.device_count() // local_world_size
        device_ids = list(range(local_rank * ngpu_per_process, (local_rank + 1) * ngpu_per_process))

        if torch.cuda.is_available() and local_rank != -1:
            torch.cuda.set_device(device_ids[0])  # device_ids[0] =local_rank if local_world_size = n_gpu per node
            device = 'cuda'
            # self.logger_info(
            #     f"[Process {os.getpid()}] world_size = {dist.get_world_size()}, "
            #     + f"rank = {dist.get_rank()}, n_gpu/process = {ngpu_per_process}, device_ids = {device_ids}"
            # )
        else:
            # self.logger_warning('Training will be using CPU!')
            device = 'cpu'
        device = torch.device(device)
        return device, device_ids
    else:
        n_gpu = torch.cuda.device_count()
        n_gpu_use = local_world_size
        if n_gpu_use > 0 and n_gpu == 0:
            # self.logger_warning("Warning: There\'s no GPU available on this machine,"
            #                     "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            # self.logger_warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
            #                     "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        list_ids = list(range(n_gpu_use))
        if n_gpu_use > 0:
            torch.cuda.set_device(list_ids[0])  # only use first available gpu as devices
            # self.logger_warning(f'Training is using GPU {list_ids[0]}!')
            device = 'cuda'
        else:
            # self.logger_warning('Training is using CPU!')
            device = 'cpu'
        device = torch.device(device)
        return device, list_ids


def entry_point(config):
    '''
    entry-point function for a single worker, distributed training
    '''

    local_world_size = config.local_world_size

    # check distributed environment cfgs
    if config.distributed:  # distributed gpu mode
        # check gpu available
        if torch.cuda.is_available():
            if torch.cuda.device_count() < local_world_size:
                raise RuntimeError(f'the number of GPU ({torch.cuda.device_count()}) is less than '
                                   f'the number of processes ({local_world_size}) running on each node')
            local_master = (config.local_rank == 0)
        else:
            raise RuntimeError('CUDA is not available, Distributed training is not supported.')
    else:  # one gpu or cpu mode
        if config.local_world_size != 1:
            raise RuntimeError('local_world_size must set be to 1, if distributed is set to false.')
        config.local_rank =  0
        local_master = True
        config.global_rank = 0

    logger = get_logger('train') if local_master else None
    if config.distributed:
        logger.info('Distributed GPU training model start...') if local_master else None
    else:
        logger.info('One GPU or CPU training mode start...') if local_master else None

    if config.distributed:
        # these are the parameters used to initialize the process group
        env_dict = {
            key: os.environ[key]
            for key in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE')
        }
        logger.info(f'[Process {os.getpid()}] Initializing process group with: {env_dict}') if local_master else None

        # init process group
        dist.init_process_group(backend='nccl', init_method='env://')
        config.global_rank = dist.get_rank()
        # info distributed training cfg
        logger.info(
            f'[Process {os.getpid()}] world_size = {dist.get_world_size()}, '
            + f'rank = {dist.get_rank()}, backend={dist.get_backend()}'
        ) if local_master else None

    # start train
    main(config, local_master, logger if local_master else None)

    # tear down the process group
    dist.destroy_process_group()

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Distributed Training')
    args.add_argument('-c', '--config', default='./experiments/visa/config.yaml', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    args.add_argument('-dist', '--distributed', action='store_true')

    args.add_argument('-local_world_size', '--local_world_size', default=1, type=int,
                      help='indices of GPUs to be available (default: all)')


    args.add_argument('-local_rank', '--local_rank', default=0, type=int,
                      help='indices of GPUs to be available (default: all)')


    args.add_argument('-global_rank', '--global_rank', default=0, type=int,
                      help='indices of GPUs to be available (default: all)')

    args_config = args.parse_args()

    entry_point(args_config)

# # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=1120 train.py --local_world_size 2 -dist
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine_measure import evaluate, train_one_epoch ##########################
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    # measure
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--fps', action='store_true')


    return parser





def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 40])
    # optimizer = torch.optim.AdamW(param_dicts, lr=0,
    #                               weight_decay=args.weight_decay)
    # #lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=args.lr,  T_up=10, gamma=1.0)
    

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'], strict=False)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        #del checkpoint['model']['detr.class_embed.weight']
        #del checkpoint['model']['detr.class_embed.bias']


        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    # AP_segm_bst = 0 ############################################       
    # AP_bbox_bst = 0###############################
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
                
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, args.throughput)
        # lr_scheduler.step()
        # if args.output_dir:
        #     checkpoint_paths = [output_dir / 'checkpoint.pth']
        #     # extra checkpoint before LR drop and every 100 epochs
        #     if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
        #         checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        #     for checkpoint_path in checkpoint_paths:
        #         utils.save_on_master({
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'args': args,
        #         }, checkpoint_path)

        # test_stats, coco_evaluator = evaluate(
        #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        # )

        # # best checkpoint 저장하자 (AP)        
        
        # if args.output_dir:
        #     checkpoint_paths = [output_dir / 'checkpoint.pth']
        #     # extra checkpoint before LR drop and every 100 epochs
        #     if args.masks:
        #         if test_stats['coco_eval_masks'][0] > AP_segm_bst:
        #             AP_segm_bst = test_stats['coco_eval_masks'][0]
        #             AP_segm_bst_s = test_stats['coco_eval_masks'][3]
        #             AP_segm_bst_m = test_stats['coco_eval_masks'][4]
        #             AP_segm_bst_l = test_stats['coco_eval_masks'][5]

        #             AP_box_bst = test_stats['coco_eval_bbox'][0]
        #             AP_box_bst_s = test_stats['coco_eval_bbox'][3]
        #             AP_box_bst_m = test_stats['coco_eval_bbox'][4]
        #             AP_box_bst_l = test_stats['coco_eval_bbox'][5]


        #             checkpoint_paths.append(output_dir / f'checkpoint_best_segm.pth')
        #             #print(output_dir)
        #             f = open('./' + str(output_dir) + '/best_segm_epoch', 'w')
        #             data = f"cuurent best segm AP : {epoch}epoch AP={AP_segm_bst} AP_small={AP_segm_bst_s} AP_medium={AP_segm_bst_m} AP_large={AP_segm_bst_l} \n \
        #                 bbox AP : AP={AP_box_bst} AP_small={AP_box_bst_s} AP_medium={AP_box_bst_m} AP_large={AP_box_bst_l}"
        #             f.write(data)
        #             f.close()
        #     else:
        #         if test_stats['coco_eval_bbox'][0] > AP_bbox_bst:
        #             AP_box_bst = test_stats['coco_eval_bbox'][0]
        #             AP_box_bst_s = test_stats['coco_eval_bbox'][3]
        #             AP_box_bst_m = test_stats['coco_eval_bbox'][4]
        #             AP_box_bst_l = test_stats['coco_eval_bbox'][5]


        #             checkpoint_paths.append(output_dir / f'checkpoint_best_box.pth')
        #             #print(output_dir)
        #             f = open('./' + str(output_dir) + '/best_segm_epoch', 'w')
        #             data = f"cuurent best box AP : AP={AP_box_bst} AP_small={AP_box_bst_s} AP_medium={AP_box_bst_m} AP_large={AP_box_bst_l}"
        #             f.write(data)
        #             f.close()
        #     # if test_stats['coco_eval_bbox'][0] > AP_bbox_bst:
        #     #     AP_bbox_bst = test_stats['coco_eval_masks'][0]
        #     #     checkpoint_paths.append(output_dir / f'checkpoint_best_bbox.pth')
        #     for checkpoint_path in checkpoint_paths:
        #         utils.save_on_master({
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'args': args,
        #         }, checkpoint_path)








        
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch,
        #              'n_parameters': n_parameters}

        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

        #     # for evaluation logs
        #     if coco_evaluator is not None:
        #         (output_dir / 'eval').mkdir(exist_ok=True)
        #         if "bbox" in coco_evaluator.coco_eval:
        #             filenames = ['latest.pth']
        #             if epoch % 50 == 0:
        #                 filenames.append(f'{epoch:03}.pth')
        #             for name in filenames:
        #                 torch.save(coco_evaluator.coco_eval["bbox"].eval,
        #                            output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    f = open('./' + str(output_dir) + '/training_time', 'w')
    data1 = f"training time : {total_time_str}\n"
    f.write(data1)
    f.close()


    if args.fps:
        fps = compute_fps(model, dataset_val, num_iters=300, batch_size=1)
        print(fps)






from util.misc import nested_tensor_from_tensor_list
from tqdm import tqdm
@torch.no_grad()
def measure_average_inference_time(model, inputs, num_iters=100, warm_iters=5):
    ts = []
    # note that warm-up iters. are excluded from the total iters.
    for iter_ in tqdm(range(warm_iters + num_iters)):
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        model(inputs)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        if iter_ >= warm_iters:
          ts.append(t)
    return sum(ts) / len(ts)

    
@torch.no_grad()
def compute_fps(model, dataset, num_iters=300, warm_iters=5, batch_size=4):
    print(f"computing fps.. (num_iters={num_iters}, batch_size={batch_size}) "
          f"warm_iters={warm_iters}, batch_size={batch_size}]")
    assert num_iters > 0 and warm_iters >= 0 and batch_size > 0
    model.cuda()
    model.eval()
    inputs = nested_tensor_from_tensor_list(
        [dataset.__getitem__(0)[0].cuda() for _ in range(batch_size)])
    t = measure_average_inference_time(model, inputs, num_iters, warm_iters)
    model.train()
    print(f"FPS: {1.0 / t * batch_size}")  
    return 1.0 / t * batch_size



######## lr
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr












if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

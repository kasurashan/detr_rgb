# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from PIL import Image
# from torchvision.utils import draw_segmentation_masks # torchvision verseion 낮음
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, dataset_val_original):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    print(dataset_val_original)
    test_iter_original0 = iter(dataset_val_original)








    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        img_original, img_original_ann = next(test_iter_original0)    # (Pil이미지, {annotation 딕셔너리(정답)})에서 첫번째 인덱스 선택
        outputs = model(samples)    # [{annotation딕셔너리(예측))}]

        image_id = img_original_ann['image_id']

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            #print(results[0]['masks'].shape)   # [100, 1, 480, 640], 각 성분은 0 또는 1의 값을 가짐
            #print(results[0]['scores'])   # 0~1사이 100개의 score값이 나옴 여기서 threshold 적용하면 될듯
            #print(torch.sum(results[0]['masks']))
            #print(results) ################[{'scores': tensor  'labels' ~ , 'boxes', 'masks'}, batch에따라 더 있을수도{~~~~}]

            #print(img_original_ann)
            segm_masks = results[0]['masks']   # [100, 1, 480, 640],
            # to_pil = transforms.ToPILImage()
            # segm_mask = to_pil(segm_mask[i])   #텐서를 pil img로 만들고
            # composite_image = Image.new('RGBA', img_original.size)
            # composite_image.paste(img_original, (0, 0))
            # composite_image.paste(segm_mask, (0, 0), mask=segm_mask)

            scores = results[0]['scores']
            labels = results[0]['labels']
            boxes = results[0]['boxes']

            idx = torch.nonzero(scores>0.1)   # score가 0.8보다 큰 인덱스들을 구하기

            
            #print(np.array(img_original).shape)  # [h, w, 3]
                        
            img_original = cv2.cvtColor(np.array(img_original, np.uint8),cv2.COLOR_RGB2BGR)     #pil에서 cv2로
            out = img_original.copy()
            for cnt, i in enumerate(idx):   # 0.8보다 큰 마스크에 대해서만 시각화 진행
                
                i = i.item()
                segm_mask = np.array(segm_masks, np.uint8)[i][0]

                label = labels[i]
                box = boxes[i]
                box = [box[0].item(), box[1].item(), box[2].item(), box[3].item()]

                
                lx, ly, rx, ry = map(int, box)

                #color = np.array([0, 10*label ,0], dtype='uint8') # label별로 색깔같도록
                
                color = np.array([255, 0 ,0], dtype='uint8')   # 토큰마다 색깔 다르게
                
                masked_img = np.where(segm_mask[...,None], color, img_original)
                out = cv2.addWeighted(img_original, 0.5, masked_img, 0.5,0)
                
                out = cv2.rectangle(out, (lx,ly), (rx, ry),  (255,0,0), 3)

                sc = round(scores[i].item(), 4)
                out = cv2.putText(out, 'score : ' + str(sc), (lx, ly+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,30,255), 2)

                
                try:
                    os.makedirs("./mask_img/img" + str(image_id.item()))
                except:
                    pass

                cv2.imwrite('./mask_img/img' +   str(image_id.item()) + '/img' + str(image_id.item()) + '_box'+ str(cnt) + '.png', out)

            
            

            
            # draw_segmentation_masks(img_original, masks=segm_mask) #torchvision 버전이 낮아서  안된다


            


            

            #composite_image.save('./mask_img/test.png')

        #img_original.save('./test.png')
            

    return results

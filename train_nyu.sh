CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 \
--use_env main.py \
--masks \
--epochs 50 \
--lr_drop 15 \
--seed 40 \
--coco_path ../../datasets/nyuv2/  \
--resume ./detr-r50-panoptic-00ce5173.pth \
--output_dir ./output/rgb_nyu_seed40teest


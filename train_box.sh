# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --master_port=44440 \
# --use_env main.py \
# --masks \
# --epochs 200 \
# --lr_drop 130 \
# --coco_path ../../datasets/boxdata/BOX_DATA/  \
# --output_dir ./output/tete \
# --resume ./detr-r50-panoptic-00ce5173.pth


CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 \
--use_env main.py \
--epochs 50  \
--lr_drop 15 \
--coco_path ../../datasets/nyuv2/  \
--resume ./detr-r50-e632da11.pth \
--output_dir ./output/RGB_nyu_50ep_detection
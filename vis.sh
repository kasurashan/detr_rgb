CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=1 \
--use_env main_vis.py \
--epochs 1 \
--masks \
--eval \
--batch_size 1 \
--coco_path ../../datasets/boxdata/BOX_DATA \
--resume /root/datasets/output_detr/RGB_box_200ep_130drop/checkpoint_best_segm.pth \
--output_dir ./output/vis
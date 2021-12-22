#!/bin/bash
source /nfs/hpc/share/nguyenkh/miniconda3/bin/activate
conda activate deocclusion

DATA="data/COCOA"


model=boundary_no_rgb_dcnv2
echo $model
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
    --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
    --order-method "ours" \
    --amodal-method "ours" \
    --order-th 0.1 \
    --amodal-th 0.2 \
    --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
    --image-root $DATA/val2014 \
    --test-num -1 \
    --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=cnp
# model=std
# model=boundary_no_rgb
# model=boundary_masked
# model=default

# model=predict_order_no_rgb
# gpu
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# gpu2
# model=predict_order_no_rgb
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=predict_order_matting
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# gpu3

# model=predict_order_matting
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# gpu4

# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

###########################

# model=default
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=default
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json




# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


###########################

# # for model in std std_no_rgb default default_no_rgb
# for model in std_no_rgb_mumford_shah
# do
#     # if [ $model == std ]; then 
#     #     epoch=34000
#     # else 
#     #     epoch=56000
#     # fi
#     # for epoch in 10000 20000 30000 40000 50000 56000
#     for epoch in 56000
#     do
#         # for method in "ours_nog" "ours"
#         for method in "ours"
#         do
#             # for th in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9
#             for th in `seq 0.1 0.05 0.9`
#             do 
#                 echo ------------- $model $method $epoch $th -----------------
#                 CUDA_VISIBLE_DEVICES=0 \
#                 python tools/test.py \
#                     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#                     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#                     --order-method "ours" \
#                     --amodal-method $method \
#                     --order-th $th \
#                     --amodal-th $th \
#                     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#                     --image-root $DATA/val2014 \
#                     --test-num -1 \
#                     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json
#             done
#         done
#     done
# done


# model=std_no_rgb_mumford_shah
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_mumford_shah
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_occluded_only
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_gaussian_boundary
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_std_no_rgb_gaussian_boundary_reg/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_gaussian_boundary
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_std_no_rgb_gaussian_boundary_reg/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_default_no_rgb_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_default_no_rgb_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_exponential
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_exponential
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


model=boundary_no_rgb
echo $model
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
    --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_8/checkpoints/ckpt_iter_56000.pth.tar"\
    --order-method "ours" \
    --amodal-method "ours" \
    --order-th 0.1 \
    --amodal-th 0.2 \
    --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
    --image-root $DATA/val2014 \
    --test-num -1 \
    --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=boundary_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json



# model=std_no_rgb_mumford_shah_mse
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=std_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json



# model=boundary_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_width_5/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "released/COCOA_pcnet_m.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=default
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_gaussian
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_exponential
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_gaussian
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_exponential
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json
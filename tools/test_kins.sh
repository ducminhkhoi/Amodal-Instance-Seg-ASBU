#!/bin/bash
source /nfs/hpc/share/nguyenkh/miniconda3/bin/activate
conda activate deocclusion

DATA="data/KINS"

# model=default_no_rgb

# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "released/KINS_pcnet_m.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/KINS/instances_train.json" \
#     --image-root $DATA/training/image_2 \
#     --test-num -1 \
#     --output experiments/KINS/pcnet_m_$model/amodal_results/amodalcomp_train_ours.json
    
# --output $DATA/amodal_results/amodalcomp_val_ours.json


# model=default_no_rgb
# epoch=32000

# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root $DATA/testing/image_2 \
#     --test-num -1 \
#     --output experiments/KINS/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json

# model=default_no_rgb_new_dataset
# epoch=32000

# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/KINS/update_test_2020.json" \
#     --image-root $DATA/testing/image_2 \
#     --test-num -1 \
#     --output experiments/KINS/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json


# model=boundary_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/LVIS/pcnet_m_boundary_no_rgb/checkpoints/ckpt_iter_168000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.5 \
#     --amodal-th 0.3 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root $DATA/testing/image_2 \
#     --test-num -1 \
#     --output experiments/KINS/pcnet_m_$model/amodal_results/amodalcomp_val_LVIS_ours.json

# model=boundary_no_rgb_new_dataset
# epoch=32000

# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.5 \
#     --amodal-th 0.3 \
#     --annotation "data/KINS/update_test_2020.json" \
#     --image-root $DATA/testing/image_2 \
#     --test-num -1 \
#     --output experiments/KINS/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json



# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "released/KINS_pcnet_m.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root $DATA/testing/image_2 \
#     --test-num -1 \
#     --output $DATA/amodal_results/amodalcomp_val_ours.json

# model=boundary_no_rgb
# model=default_no_rgb
# epoch=32000

# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root $DATA/testing/image_2 \
#     --test-num -1 \
#     --output $DATA/amodal_results/amodalcomp_val_ours.json

# model=boundary_no_rgb
# epoch=18000

# # for th in `seq 0 0.05 0.25`
# for th in `seq 0.1 0.05 0.9`
# # for th in `seq 0.55 0.05 0.9`
# do
#     echo $th
#     CUDA_VISIBLE_DEVICES=0 \
#     python tools/test.py \
#         --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#         --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#         --order-method "ours" \
#         --amodal-method "ours" \
#         --order-th $th \
#         --amodal-th 0.3 \
#         --annotation "data/KINS/instances_val.json" \
#         --image-root $DATA/testing/image_2 \
#         --test-num -1 \
#         --output $DATA/amodal_results/amodalcomp_val_ours.json

# done

# model=std_no_rgb_exponential
# epoch=18000

# # for th in `seq 0 0.05 0.25`
# for epoch in `seq 2000 2000 16000`
# # for th in `seq 0.3 0.05 0.5`
# do
#     echo $model $epoch
#     CUDA_VISIBLE_DEVICES=0 \
#     python tools/test.py \
#         --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#         --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#         --order-method "ours" \
#         --amodal-method "ours" \
#         --order-th 0.75 \
#         --amodal-th 0.75 \
#         --annotation "data/KINS/instances_val.json" \
#         --image-root $DATA/testing/image_2 \
#         --test-num -1 \
#         --output $DATA/amodal_results/amodalcomp_val_ours.json

# done

# model=boundary_no_rgb_exponential

# # for th in `seq 0 0.05 0.25`
# for epoch in `seq 4000 4000 32000`
# # for th in `seq 0.3 0.05 0.5`
# do
#     echo $model $epoch
#     CUDA_VISIBLE_DEVICES=0 \
#     python tools/test.py \
#         --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#         --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#         --order-method "ours" \
#         --amodal-method "ours" \
#         --order-th 0.75 \
#         --amodal-th 0.75 \
#         --annotation "data/KINS/instances_val.json" \
#         --image-root $DATA/testing/image_2 \
#         --test-num -1 \
#         --output $DATA/amodal_results/amodalcomp_val_ours.json

# done

# model=std_no_rgb_gaussian
# epoch=16000

# # for th in `seq 0 0.05 0.25`
# for th in `seq 0.1 0.05 0.9`
# # for th in `seq 0.55 0.05 0.9`
# do
#     echo $th
#     CUDA_VISIBLE_DEVICES=0 \
#     python tools/test.py \
#         --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#         --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#         --order-method "ours" \
#         --amodal-method "ours" \
#         --order-th $th \
#         --amodal-th 0.3 \
#         --annotation "data/KINS/instances_val.json" \
#         --image-root $DATA/testing/image_2 \
#         --test-num -1 \
#         --output $DATA/amodal_results/amodalcomp_val_ours.json

# done

# model=std_no_rgb_gaussian
# epoch=56000

# model=std_no_rgb_gaussian
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_16000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.4 \
#     --amodal-th 0.3 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root data/KINS/testing/image_2 \
#     --test-num -1 \
#     --output experiments/KINS/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json

model=std_no_rgb_cross_entropy_gaussian
echo $model
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --config experiments/KINS/pcnet_m/config_train_$model.yaml \
    --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_4000.pth.tar"\
    --order-method "ours" \
    --amodal-method "ours" \
    --order-th 0.4 \
    --amodal-th 0.3 \
    --annotation "data/KINS/instances_val.json" \
    --image-root data/KINS/testing/image_2 \
    --test-num -1 \
    --output experiments/KINS/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json


# model=boundary_no_rgb_gaussian
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_28000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.4 \
#     --amodal-th 0.3 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root data/KINS/testing/image_2 \
#     --test-num -1 \
#     --output experiments/KINS/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json

# model=boundary_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_18000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.5 \
#     --amodal-th 0.3 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root data/KINS/testing/image_2 \
#     --test-num -1 \
#     --output experiments/KINS/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json


# echo $th
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.5 \
#     --amodal-th 0.3 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root $DATA/testing/image_2 \
#     --test-num -1 \
#     --output $DATA/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_mumford_shah
# epoch=56000

# echo $th
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.4 \
#     --amodal-th 0.3 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root $DATA/testing/image_2 \
#     --test-num -1 \
#     --output $DATA/amodal_results/amodalcomp_val_ours.json


# model=std_no_rgb_exponential
# epoch=56000

# echo $th
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.75 \
#     --amodal-th 0.75 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root $DATA/testing/image_2 \
#     --test-num -1 \
#     --output $DATA/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb
# epoch=56000

# echo $th
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.5 \
#     --amodal-th 0.3 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root $DATA/testing/image_2 \
#     --test-num -1 \
#     --output $DATA/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_input_size_64

# for epoch in {18000..32000..2000}
# do 
#     echo $model $epoch 
#     CUDA_VISIBLE_DEVICES=0 \
#     python tools/test.py \
#         --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#         --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#         --order-method "ours" \
#         --amodal-method "ours" \
#         --order-th 0.1 \
#         --amodal-th 0.2 \
#         --annotation "data/KINS/instances_val.json" \
#         --image-root $DATA/testing/image_2 \
#         --test-num -1 \
#         --output $DATA/amodal_results/amodalcomp_val_ours.json
# done


# model=boundary_no_rgb_input_size_64

# for epoch in {2000..16000..2000}
# do 
#     echo $model $epoch 
#     CUDA_VISIBLE_DEVICES=0 \
#     python tools/test.py \
#         --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#         --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#         --order-method "ours" \
#         --amodal-method "ours" \
#         --order-th 0.1 \
#         --amodal-th 0.2 \
#         --annotation "data/KINS/instances_val.json" \
#         --image-root $DATA/testing/image_2 \
#         --test-num -1 \
#         --output $DATA/amodal_results/amodalcomp_val_ours.json
# done

# model=boundary_no_rgb_gaussian

# for epoch in {28000..32000..6000}
# do 
#     echo $model $epoch 
#     CUDA_VISIBLE_DEVICES=0 \
#     python tools/test.py \
#         --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#         --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_$epoch.pth.tar"\
#         --order-method "ours" \
#         --amodal-method "ours" \
#         --order-th 0.1 \
#         --amodal-th 0.2 \
#         --annotation "data/KINS/instances_val.json" \
#         --image-root $DATA/testing/image_2 \
#         --test-num -1 \
#         --output $DATA/amodal_results/amodalcomp_val_ours.json
# done


# model=std_no_rgb_gaussian

# for epoch in {2000..32000..4000}
# do 
#     echo $model $epoch 
#     CUDA_VISIBLE_DEVICES=0 \
#     python tools/test.py \
#         --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#         --load-model "experiments/KINS/pcnet_m_std_no_rgb_gaussian_no_category/checkpoints/ckpt_iter_$epoch.pth.tar"\
#         --order-method "ours" \
#         --amodal-method "ours" \
#         --order-th 0.4 \
#         --amodal-th 0.3 \
#         --annotation "data/KINS/instances_val.json" \
#         --image-root $DATA/testing/image_2 \
#         --test-num -1 \
#         --output $DATA/amodal_results/amodalcomp_val_ours.json
# done


# model=std_no_rgb_gaussian

# for epoch in {4000..32000..4000}
# do 
#     echo $model $epoch 
#     CUDA_VISIBLE_DEVICES=0 \
#     python tools/test.py \
#         --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#         --load-model "experiments/KINS/pcnet_m_std_no_rgb_gaussian_no_category/checkpoints/ckpt_iter_$epoch.pth.tar"\
#         --order-method "ours" \
#         --amodal-method "ours" \
#         --order-th 0.4 \
#         --amodal-th 0.3 \
#         --annotation "data/KINS/instances_val.json" \
#         --image-root $DATA/testing/image_2 \
#         --test-num -1 \
#         --output $DATA/amodal_results/amodalcomp_val_ours.json
# done

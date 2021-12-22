source /nfs/hpc/share/nguyenkh/miniconda3/bin/activate
conda activate deocclusion

# model=boundary_no_rgb_mumford_shah
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/vis_qual.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_mumford_shah_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.5 \
#     --amodal-th 0.5 \
#     --annotation data/COCOA/annotations/COCO_amodal_val2014.json \
#     --image-root data/COCOA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/qual.json

model=boundary_no_rgb_gaussian
echo $model
CUDA_VISIBLE_DEVICES=0 \
python tools/vis_qual.py \
    --config experiments/KINS/pcnet_m/config_train_$model.yaml \
    --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_32000.pth.tar"\
    --order-method "ours" \
    --amodal-method "ours" \
    --order-th 0.4 \
    --amodal-th 0.3 \
    --annotation "data/KINS/instances_val.json" \
    --image-root data/KINS/testing/image_2 \
    --test-num -1 \
    --output experiments/KINS/pcnet_m_$model/amodal_results/qual.json

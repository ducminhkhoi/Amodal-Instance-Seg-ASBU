source /nfs/hpc/share/nguyenkh/miniconda3/bin/activate
conda activate deocclusion

DATA="data/COCOA"

model=boundary_no_rgb
echo $model
CUDA_VISIBLE_DEVICES=0 \
python tools/vis_mutual.py \
    --config experiments/LVIS/pcnet_m/config_train_$model.yaml \
    --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_8/checkpoints/ckpt_iter_56000.pth.tar"\
    --order-method "ours" \
    --amodal-method "ours_nog" \
    --order-th 0.1 \
    --amodal-th 0.2 \
    --annotation "mutual/mutual_coco.json" \
    --image-root mutual \
    --dilate_kernel 5 \
    --scale 7 \
    --test-num -1 \
    --output experiments/COCOA/pcnet_m_$model/amodal_results/mutual.json

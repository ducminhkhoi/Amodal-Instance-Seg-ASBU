source /nfs/hpc/share/nguyenkh/miniconda3/bin/activate
conda activate deocclusion


dataset=COCOA # change the dataset name here COCOA or KINS

work_path=experiments/$dataset/pcnet_m

mode=std_no_rgb_gaussian

python -m torch.distributed.launch --nproc_per_node=1 main.py \
    --config $work_path/config_train_$mode.yaml --launcher pytorch \
    --load-iter 56000 --validate --exp_path experiments/$dataset/pcnet_m_std_no_rgb_gaussian

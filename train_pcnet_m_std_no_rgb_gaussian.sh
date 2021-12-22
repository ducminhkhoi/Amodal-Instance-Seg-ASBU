source /nfs/hpc/share/nguyenkh/miniconda3/bin/activate
conda activate deocclusion

num_proc=2

dataset=COCOA # change the dataset name here COCOA or KINS

work_path=experiments/$dataset/pcnet_m

OMP_NUM_THREADS=$num_proc python -m torch.distributed.launch --nproc_per_node=$num_proc --master_port=9918 main.py --config $work_path/config_train_std_no_rgb_gaussian.yaml --launcher pytorch --exp_path experiments/$dataset/pcnet_m_std_no_rgb_gaussian



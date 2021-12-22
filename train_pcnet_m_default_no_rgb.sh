source /nfs/hpc/share/nguyenkh/miniconda3/bin/activate
conda activate deocclusion

export TORCH_HOME=/nfs/hpc/share/nguyenkh/torch

dataset=COCOA # change the dataset name here COCOA or KINS

num_proc=2

work_path=experiments/$dataset/pcnet_m

OMP_NUM_THREADS=$num_proc python -m torch.distributed.launch --nproc_per_node=$num_proc --master_port=9917 main.py --config $work_path/config_train_default_no_rgb.yaml --launcher pytorch --exp_path experiments/$dataset/pcnet_m_default_no_rgb


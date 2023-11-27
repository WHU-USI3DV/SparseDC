#/usr/bin
cd ../
EXP=$1
NAME=$2
CKPT_PATH=$3
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXP ckpt_path=$CKPT_PATH num_sample=shift_grid task_name=${NAME}_shift_grid_sample ++trainer.deterministic=True $5
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXP ckpt_path=$CKPT_PATH num_sample=uneven_density task_name=${NAME}_uneven_density ++trainer.deterministic=True $5
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXP ckpt_path=$CKPT_PATH num_sample=holes task_name=${NAME}_holes_sample ++trainer.deterministic=True $5
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXP ckpt_path=$CKPT_PATH num_sample=500 task_name=${NAME}_500_sample ++trainer.deterministic=True $5
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXP ckpt_path=$CKPT_PATH num_sample=200 task_name=${NAME}_200_sample ++trainer.deterministic=True $5
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXP ckpt_path=$CKPT_PATH num_sample=50 task_name=${NAME}_50_sample ++trainer.deterministic=True $5
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXP ckpt_path=$CKPT_PATH num_sample=10 task_name=${NAME}_10_sample ++trainer.deterministic=True $5
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXP ckpt_path=$CKPT_PATH num_sample=5 task_name=${NAME}_5_sample ++trainer.deterministic=True $5
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXP ckpt_path=$CKPT_PATH num_sample=keypoints_orb task_name=${NAME}_ORB ++trainer.deterministic=True $5

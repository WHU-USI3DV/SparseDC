#/usr/bin
cd ../
NAME=$1
EXPERIMENTS=$2
CKPT_PATH=$3
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXPERIMENTS ckpt_path=$CKPT_PATH task_name=${NAME}_lines64 ++data.args.num_lines=lines64
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXPERIMENTS ckpt_path=$CKPT_PATH task_name=${NAME}_lines32 ++data.args.num_lines=lines32
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXPERIMENTS ckpt_path=$CKPT_PATH task_name=${NAME}_lines16 ++data.args.num_lines=lines16
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXPERIMENTS ckpt_path=$CKPT_PATH task_name=${NAME}_lines8 ++data.args.num_lines=lines8
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$EXPERIMENTS ckpt_path=$CKPT_PATH ++data.args.num_lines=lines4 task_name=${NAME}_lines4
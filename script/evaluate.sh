#!/bin/bash -l
#SBATCH --job-name=mattergen_finetune_multi                      # 作业名称
#SBATCH --output=/home/uoegr7/mattergen/mattergen/mattergen_logs/evaluation.%j.out  # 将标准输出文件存放在指定文件夹
#SBATCH --error=/home/uoegr7/mattergen/mattergen/mattergen_logs/evaluation.%j.err   # 将错误输出文件存放在指定文件夹
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=gpu                             # 指定GPU分区
#SBATCH --gres=gpu:1                                # 请求1块GPU

## 加载必要模块 ##
module purge
module load cuda/11.8  # 加载 CUDA 模块 (根据实际情况选择 CUDA 版本)

## 激活虚拟环境 ##
source /home/uoegr7/mattergen/mattergen/.venv/bin/activate  # 激活虚拟环境

# 2. 设置结果保存的路径
export RESULTS_PATH="results/finetuned_generation_7/"

# 3. 运行生成命令
mattergen-evaluate --structures_path=$RESULTS_PATH \
    --relax=True \
    --structure_matcher='disordered' \
    --potential_load_path="/home/uoegr7/mattergen/mattergen/evaluation/mattersim-v1.0.0-5M.pth" \
    --save_as="$RESULTS_PATH/metrics.json" \
    --structures_output_path="$RESULTS_PATH/relaxed_structures.extxyz"
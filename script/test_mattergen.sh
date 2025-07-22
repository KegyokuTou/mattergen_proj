#!/bin/bash -l
#SBATCH --job-name=mattergen_gpu                      # 作业名称
#SBATCH --output=/home/uoegr7/mattergen/mattergen/mattergen_logs/mattergen_gpu.%j.out  # 将标准输出文件存放在指定文件夹
#SBATCH --error=/home/uoegr7/mattergen/mattergen/mattergen_logs/mattergen_gpu.%j.err   # 将错误输出文件存放在指定文件夹
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=gpu                             # 指定GPU分区
#SBATCH --gres=gpu:1                                # 请求1块GPU

## 加载必要模块 ##
module purge
module load cuda/11.8  # 加载 CUDA 模块 (根据实际情况选择 CUDA 版本)

## 激活虚拟环境 ##
source /home/uoegr7/mattergen/mattergen/.venv/bin/activate  # 激活虚拟环境 (请修改为您的虚拟环境路径)

## 设置环境变量 (根据您的实际情况修改) ##
export MODEL_NAME=mattergen_base
export RESULTS_PATH=/home/uoegr7/mattergen/mattergen/results/  # 修改为您的结果输出路径

## 运行 MatterGen (根据您的实际需求修改命令) ##
mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME --batch_size=16 --num_batches 1

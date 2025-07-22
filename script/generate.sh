#!/bin/bash -l
#SBATCH --job-name=mattergen_finetune_multi                      # 作业名称
#SBATCH --output=/home/uoegr7/mattergen/mattergen/mattergen_logs/mattergen_generate.%j.out  # 将标准输出文件存放在指定文件夹
#SBATCH --error=/home/uoegr7/mattergen/mattergen/mattergen_logs/mattergen_generate.%j.err   # 将错误输出文件存放在指定文件夹
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=gpu                             # 指定GPU分区
#SBATCH --gres=gpu:1                                # 请求1块GPU

## 加载必要模块 ##
module purge
module load cuda/11.8  # 加载 CUDA 模块 (根据实际情况选择 CUDA 版本)

## 激活虚拟环境 ##
source /home/uoegr7/mattergen/mattergen/.venv/bin/activate  # 激活虚拟环境

export MODEL_PATH="/home/uoegr7/mattergen/mattergen/outputs/singlerun/2025-07-17/14-48-18"

# 2. 设置结果保存的路径
mkdir -p results/finetuned_generation_8/
export RESULTS_PATH="results/finetuned_generation_8/"

# 3. 运行生成命令
mattergen-generate $RESULTS_PATH \
    --model_path=$MODEL_PATH \
    --batch_size=192 \
    --num_batches 3 \
    --properties_to_condition_on="{'energy_above_hull': 0.0 }" \
    --diffusion_guidance_factor = 2
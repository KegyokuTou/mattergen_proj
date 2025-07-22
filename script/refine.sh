#!/bin/bash -l
#SBATCH --job-name=mattergen_finetune_multi                      # 作业名称
#SBATCH --output=/home/uoegr7/mattergen/mattergen/mattergen_logs/mattergen_finetune_multi.%j.out  # 将标准输出文件存放在指定文件夹
#SBATCH --error=/home/uoegr7/mattergen/mattergen/mattergen_logs/mattergen_finetune_multi.%j.err   # 将错误输出文件存放在指定文件夹
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=gpu4-80GB                             # 指定GPU分区
#SBATCH --gres=gpu:1                                # 请求1块GPU

## 加载必要模块 ##
module purge
module load cuda/11.8  # 加载 CUDA 模块 (根据实际情况选择 CUDA 版本)

## 激活虚拟环境 ##
source /home/uoegr7/mattergen/mattergen/.venv/bin/activate  # 激活虚拟环境

## 设置环境变量##

export PROPERTY1=chemical_system
export PROPERTY2=dft_band_gap 
export PROPERTY3=energy_above_hull
export MODEL_NAME=mattergen_base

## 运行 MatterGen 微调 ##

mattergen-finetune adapter.pretrained_name=$MODEL_NAME data_module=Li_refine +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY3=$PROPERTY3 ~trainer.logger data_module.properties=["$PROPERTY1","$PROPERTY2","$PROPERTY3"]
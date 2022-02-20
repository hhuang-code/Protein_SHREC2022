#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --job-name='shrec2022'
##SBATCH --exclusive
#SBATCH -p nvidia

#SBATCH --mail-type=END
#SBATCH --mail-user=hh1811@nyu.edu

module purge

source ~/.bashrc
conda activate haohuang

cd /scratch/hh1811/projects/Protein_SHREC2022/pytorch

TRAIN_CODE=train.py

dataset=shrec2022
exp_name=pointtransformer
setting=binary_c_$(date +"%Y%m%d")
#setting=multiple_$(date +"%Y%m%d")
exp_dir=exp/${dataset}/${exp_name}/${setting}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}_${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
cp tool/train.sh tool/${TRAIN_CODE} ${config} ${exp_dir}

now=$(date +"%Y%m%d_%H%M%S")

$(which python) ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  resume ${exp_dir}/model/model_last.pth 2>&1 | tee ${exp_dir}/train-$now.log
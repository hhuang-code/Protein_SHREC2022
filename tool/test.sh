#!/bin/sh

module purge

source ~/.bashrc
conda activate haohuang

cd /scratch/hh1811/projects/Protein_SHREC2022/pytorch

TEST_CODE=test.py

dataset=shrec2022
exp_name=pointtransformer
setting=binary_c_20220218
exp_dir=exp/${dataset}/${exp_name}/${setting}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}_${exp_name}.yaml

mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

now=$(date +"%Y%m%d_%H%M%S")
cp ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}

#$(which python) -u ${exp_dir}/${TEST_CODE} \
#  --config=${config} \
#  save_folder ${result_dir}/best \
#  model_path ${model_dir}/model_best.pth \
#  2>&1 | tee ${exp_dir}/test_best-$now.log

$(which python) -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_folder ${result_dir}/last \
  model_path ${model_dir}/model_last.pth #2>&1 | tee ${exp_dir}/test_last-$now.log


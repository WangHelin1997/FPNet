#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='/home/helin.wang/data/ESC-50'

# You need to modify this path to your workspace to store features and models
WORKSPACE='/home/helin.wang/code/workspace'

# Hyper-parameters
GPU_ID=1
MODEL_TYPE='Cnn_9layers_AvgPooling_1D'
BATCH_SIZE=64

############ Train and validate on dataset ############
# Calculate feature
#python util/feature_1D.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Calculate scalar
#python util/feature_1D.py calculate_scalar --workspace=$WORKSPACE

# Tarin
CUDA_VISIBLE_DEVICES=$GPU_ID python util/main_1D.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='/home/cdd/code/WaveMsNet-master/ESC-50-master'

# You need to modify this path to your workspace to store features and models
WORKSPACE='/home/cdd/code/FPNet/workspace'

# Hyper-parameters
GPU_ID=2
MODEL_TYPE='Cnns'
BATCH_SIZE=64

############ Train and validate on dataset ############
# # Calculate feature
# python util/feature_1D.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# # Tarin
CUDA_VISIBLE_DEVICES=$GPU_ID python util/main_1D.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

# Calculate feature
# python util/feature.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Tarin
# CUDA_VISIBLE_DEVICES=$GPU_ID python util/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

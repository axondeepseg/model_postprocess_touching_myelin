#!/bin/bash
#
# Training nnUNetv2 on multiple folds
#
# NOTE: This is a template script, modify it as needed
#
# Modified from the script by Naga Karthik and Jan Valosek
# located in: https://github.com/ivadomed/utilities/blob/main/scripts/run_nnunet.sh
#

config="2d"                     
dataset_id=1             
dataset_name=Dataset001_MyelinBoundarySegmentation     
nnunet_trainer="nnUNetTrainer"

# Select number of folds here
folds=(0 1 2 3 4)

echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"

nnUNetv2_plan_and_preprocess -d ${dataset_id} --verify_dataset_integrity

for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"

    # training
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_train ${dataset_id} ${config} ${fold} -tr ${nnunet_trainer}

    echo ""
    echo "-------------------------------------------"
    echo "Training completed"
    echo "-------------------------------------------"

done
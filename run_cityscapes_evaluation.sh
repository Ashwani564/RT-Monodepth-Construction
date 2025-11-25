#!/bin/bash
# Script to evaluate all RT-MonoDepth models on Cityscapes validation set

echo "=========================================="
echo "RT-MonoDepth Cityscapes Evaluation"
echo "=========================================="
echo ""

# List of all model variants
models=(
    "full sh_640_192"
    "full s_640_192"
    "full m_640_192"
    "full ms_640_192"
    "s m_640_192"
    "s ms_640_192"
)

# Run evaluation for each model
for model_info in "${models[@]}"; do
    read -r model_type model_name <<< "$model_info"
    
    echo "=========================================="
    echo "Evaluating: ${model_type}/${model_name}"
    echo "=========================================="
    echo ""
    
    python benchmark/evaluate_depth_multi_dataset.py \
        --model_path "weights/RTMonoDepth/${model_type}/${model_name}" \
        --model_type "${model_type}" \
        --datasets cityscapes \
        --cityscapes_split val \
        --batch_size 8 \
        --num_workers 4 \
        --output_dir benchmark/results
    
    echo ""
    echo "✅ Completed: ${model_type}/${model_name}"
    echo ""
done

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="

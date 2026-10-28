#!/usr/bin/zsh

#Adapted and modified from Muennighoff, N, Yang, Z, Shi, W, Li, XL, Fei-Fei, L, Hajishirzi, H, Zettlemoyer, L, Liang, P, Candès, E & Hashimoto, T 2025, s1: Simple test-time scaling. Available from: https://github.com/simplescaling/s1 [19. September 2026] under Apache License Version 2.0, January 2004 found under /Third_Party_Licenses
# Set training parameters
lr=1e-5
epochs=5
batch_size=16
weight_decay=1e-4
train_dataset_name="Training_Data_final.json"
uid="$(date +%Y%m%d_%H%M%S)"

# Parse command-line arguments 
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;; 
        --epochs) epochs="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --train_dataset_name) train_dataset_name="$2"; shift 2 ;;
        --uid) uid="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Get node information
node_array=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nnodes=$(echo $node_array | wc -w)

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600


# Calculate gradient accumulation steps
gpu_count=$(nvidia-smi -L | wc -l)
grad_acc=4

echo "Number of nodes: $nnodes"
echo "Number of GPUs per node: $gpu_count"
#echo "Head node IP: $head_node_ip"
# Launch distributed training using srun
run_name="qwen_${train_dataset_name}_bs${batch_size}_lr${lr}_epoch${epochs}_wd${weight_decay}_${uid}"

module load Python
source /home/x/venvs/sft-test/bin/activate


torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    SFT/SFT/sft.py \
    --block_size=1024 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=$grad_acc \
    --train_file_path="Training/${train_dataset_name}" \
    --model_name="Qwen/Qwen2.5-14B-Instruct" \
    --warmup_ratio=0.05 \
    --report_to="none" \
    --bf16=True \
    --eval_strategy="no" \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="SFT/SFT/fsdp_config_qwen_cpu.json" \
    --gradient_checkpointing=True \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="/hpcwork/x/${run_name}.out" \
    --push_to_hub=false \
    --save_only_model=True \
    --num_train_epochs=${epochs} \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
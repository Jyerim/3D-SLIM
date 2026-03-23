which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((54000 + $RANDOM % 10000))
export MASTER_ADDR=localhost

epoch=3
batch_size=32
lr=5e-6
input_dim=1024 # 1024
different_lr=False
max_obj_num=100
lora_r=16
lora_alpha=16
config="config"
max_grad_norm=0.01
seed=42
llama_model_path="./llm/vicuna-7b-v1.5"
# llama_model_path="./llm/Meta-Llama-3-8B-Instruct"
# llama_model_path="./llm/Qwen2-7B-Instruct"

train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#nr3d_caption#obj_align"
val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
debug=False
if [ $debug = "True" ]; then
    enable_wandb=False
    gpu_num=2
    do_save=False
else
    enable_wandb=True
    gpu_num=2
    do_save=True
fi

### Ours (Vicuna) ###
evaluate=False
permute=True
bidirection=True
mask_type='GeoMask+InstMask'
top_k_min=2
top_k_max=10
pretrained_path=""
OUTPUT_DIR=outputs/vicuna+3DSLIM
mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=2 --master_port=23455 tasks/train.py \
    "$(dirname $0)/${config}.py" \
    output_dir "${OUTPUT_DIR}" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    pretrained_path "$pretrained_path" \
    evaluate "$evaluate" \
    wandb.enable "$enable_wandb" \
    gpu_num "$gpu_num" \
    do_save "$do_save" \
    batch_size "$batch_size" \
    train_tag "$train_tag" \
    val_tag "$val_tag" \
    model.input_dim "$input_dim" \
    model.bidirection "$bidirection" \
    optimizer.different_lr.enable "$different_lr" \
    model.max_obj_num "$max_obj_num" \
    lora.lora_r "$lora_r" \
    lora.lora_alpha "$lora_alpha" \
    optimizer.max_grad_norm "$max_grad_norm" \
    seed "$seed" \
    model.llama_model_path "$llama_model_path" \
    model.mask_type "$mask_type" \
    model.top_k_min "$top_k_min" \
    model.top_k_max "$top_k_max"
# ========================= data ==========================
anno_root = "annotations"  # annotation dir
pc_encoder = "uni3d"
segmentor = "mask3d"
version = "_nonoverlap"
train_iou_thres=0.5
overlap_iou_thres=0.9

seg_feat_file = f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt"
seg_img_feat_file = f"{anno_root}/scannet_{segmentor}_videofeats_ours.pt"
seg_train_attr_file = f"{anno_root}/scannet_{segmentor}_train_attributes{version}_{overlap_iou_thres}.pt"
seg_val_attr_file = f"{anno_root}/scannet_{segmentor}_val_attributes{version}_{overlap_iou_thres}.pt"

seg_all_feat_file = f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats_all.pt"
seg_all_img_feat_file = f"{anno_root}/scannet_{segmentor}_videofeats_all.pt"
seg_all_attr_file =f"{anno_root}/scannet_{segmentor}_all_attributes_nonoverlap_0.9.pt"

train_tag = 'scanqa'
val_tag = 'scanqa'

train_file_dict = {
    'scanrefer': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_train{version}_{overlap_iou_thres}_trainiou_{train_iou_thres}.json",
    ],
    'scan2cap': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_train{version}_{overlap_iou_thres}_trainiou_{train_iou_thres}.json",
    ],
    'nr3d_caption': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/nr3d_caption_{segmentor}_train{version}_{overlap_iou_thres}_trainiou_{train_iou_thres}.json",
    ],
    'obj_align': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/obj_align_{segmentor}_train{version}_{overlap_iou_thres}_trainiou_{train_iou_thres}.json",
    ],
    'scanqa': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanqa_train.json",
    ],
    'sqa3d': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/sqa3d_train.json",
    ],
    'multi3dref': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_train{version}_{overlap_iou_thres}_trainiou_{train_iou_thres}.json",
    ],
    'nr3d': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/nr3d_{segmentor}_train{version}_{overlap_iou_thres}_trainiou_{train_iou_thres}.json",
    ],
    'sr3d': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/sr3d_{segmentor}_train{version}_{overlap_iou_thres}_trainiou_{train_iou_thres}.json",
    ],
}

val_file_dict = {
    'scanqa': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanqa_val.json",
    ],
    'scanrefer': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_val{version}_{overlap_iou_thres}_trainiou_{train_iou_thres}.json",
    ],
    'scan2cap': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_val{version}_{overlap_iou_thres}_trainiou_{train_iou_thres}.json",
    ],
    'sqa3d': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/sqa3d_val.json",
    ],
    'sqa3d_test': [
        seg_all_feat_file,
        seg_all_img_feat_file,
        seg_all_attr_file,
        f"{anno_root}/sqa3d_test.json",
    ],
    'multi3dref': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_val{version}_{overlap_iou_thres}_trainiou_{train_iou_thres}.json",
    ],
}
box_mode="pred"
num_workers = 4
batch_size = 16
permute = True


# ========================= model ==========================
model = dict(
    llama_model_path="llm/vicuna-7b-v1.5",
    input_dim=1024,
    img_input_dim=1024,
    pos_dim=128,
    low_resource=False,
    system_path="prompts/system.txt",
    instruction_path="prompts/instruction.txt",
    max_txt_len=64,
    end_sym="</s>",
    role=("USER", "ASSISTANT"),
    use_lora=True,
    max_obj_num=100,
    bidirection=False,
    mask_type='GeoMask+InstMask',
    positional_emb=False,
    top_k_min=2,
    top_k_max=10,
)

lora = dict(
    lora_target_modules=[
      "q_proj",
      "v_proj",
      "k_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.05
)
gradnorm = dict(
    opt = "adamW",
    w_lr = 5e-5,
    w_betas=[0.9, 0.999],
    weight_decay=0.0,
    alpha=1.0
)
optimizer = dict(
    opt="adamW",
    lr=5e-6,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    scaler_enable=False,
    accum_iter=1,
    max_grad_norm=0.01,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(
        enable=False,
        module_names=["model.embed_tokens"],
        lr=[1e-6],
        wd=[0.02]
    ),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="",
)
dist_url = "env://"
device = "cuda"

# ========================= others ==========================
output_dir = "outputs/tmp"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 20
# eval_freq = 500
seed = 42

save_latest = False
do_save = True
auto_resume = True
pretrained_path = ""
img_projector_path = ""

debug=False
gpu_num=1

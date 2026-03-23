#! /bin/bash

scannet_dir="/workspace/data/scannet"
version="_nonoverlap"
segment_result_dir="None"
inst_seg_dir="/project/About_3DReason/Chat-Scene/annotations/mask3d_inst_seg"
class_label_file="annotations/scannet/scannetv2-labels.combined.tsv"
max_obj_num=100
overlap_iou_thres=0.9
train_iou_thres=0.5

processed_data_dir="/project/About_3DReason/Chat-Scene/annotations/mask3d_inst_data"
segmentor="mask3d"

python preprocess/prepare_mask3d_data.py \
    --scannet_dir "$scannet_dir" \
    --output_dir "$processed_data_dir" \
    --segment_dir "$segment_result_dir" \
    --inst_seg_dir "$inst_seg_dir" \
    --class_label_file "$class_label_file" \
    --apply_global_alignment \
    --num_workers 16 \
    --parallel

python preprocess/prepare_scannet_attributes.py \
    --scannet_dir "$scannet_dir"

python preprocess/prepare_scannet_mask3d_attributes.py \
    --scan_dir "$processed_data_dir" \
    --segmentor "$segmentor" \
    --max_inst_num "$max_obj_num" \
    --overlap_iou_thres "$overlap_iou_thres" \
    --version "$version"

python preprocess/prepare_scanrefer_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres" \
    --overlap_iou_thres "$overlap_iou_thres" \
    --max_obj_num "$max_obj_num"

python preprocess/prepare_scan2cap_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres" \
    --overlap_iou_thres "$overlap_iou_thres" \
    --max_obj_num "$max_obj_num"

python preprocess/prepare_objalign_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres" \
    --overlap_iou_thres "$overlap_iou_thres" \

python preprocess/prepare_nr3dcaption_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres" \
    --overlap_iou_thres "$overlap_iou_thres" \

python preprocess/prepare_multi3dref_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres" \
    --overlap_iou_thres "$overlap_iou_thres" \

python preprocess/prepare_scanqa_annos.py

python preprocess/prepare_sqa3d_annos.py

# python preprocess/prepare_nr3d_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --max_obj_num "$max_obj_num" \
#     --overlap_iou_thres "$overlap_iou_thres"

# python preprocess/prepare_sr3d_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --max_obj_num "$max_obj_num" \
#     --overlap_iou_thres "$overlap_iou_thres"
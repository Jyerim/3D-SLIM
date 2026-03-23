import torch
import json
import os
import glob
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--scan_dir', required=True, type=str,
                    help='the path of the directory to be saved preprocessed scans')
parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--max_inst_num', required=True, type=int)
parser.add_argument('--overlap_iou_thres', required=True, type=float)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--islayout', action="store_true")
args = parser.parse_args()

### scannet train/val set
def compute_3d_iou(seg_a, seg_b):
    set_a, set_b = set(seg_a), set(seg_b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if len(union) > 0 else 0.0

# train split일 때 GT 세그먼트를 미리 로드
gt_segs_all = {}
if '_nonoverlap' in args.version:
    scannet_attribute_file = f"annotations/scannet_train_attributes.pt"
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')
        
for split in ["train", "val"]:
    scan_dir = os.path.join(args.scan_dir, 'pcd_all')
    output_dir = "annotations"
    split_path = f"annotations/scannet/scannetv2_{split}.txt"

    scan_ids = [line.strip() for line in open(split_path).readlines()]

    scan_ids = sorted(scan_ids)
    # print(scan_ids)

    scans = {}
    group_infos = {}
    for scan_id in tqdm(scan_ids):
        pcd_path = os.path.join(scan_dir, f"{scan_id}.pth")
        if not os.path.exists(pcd_path):
            print('skip', scan_id)
            continue
        points, colors, instance_class_labels, instance_segids = torch.load(pcd_path)
        inst_locs = []
        valid_mask = []
        num_insts = len(instance_class_labels)
        print(num_insts, args.max_inst_num)
        max_instances = min(num_insts, args.max_inst_num)
        for i in range(max_instances):
            inst_mask = instance_segids[i]
            pc = points[inst_mask]
            if len(pc) == 0:
                print(scan_id, i, 'empty bbox')
                inst_locs.append(np.zeros(6, ).astype(np.float32))
                valid_mask.append(0)
                continue
            size = pc.max(0) - pc.min(0)
            center = (pc.max(0) + pc.min(0)) / 2
            inst_locs.append(np.concatenate([center, size], 0))
            valid_mask.append(1)
            
        while len(inst_locs) < args.max_inst_num:
            inst_locs.append(np.zeros(6, dtype=np.float32))
            instance_class_labels.append(-1)
            valid_mask.append(0)
            
        inst_locs = torch.tensor(np.stack(inst_locs, 0), dtype=torch.float32)
        
        # 클러스터링 방식: IoU가 0.9 이상인 proposal들을 그룹화하여 대표 proposal만 valid로 선택
        if '_nonoverlap' in args.version:
            
            groups = []  # 각 그룹은 proposal 인덱스 리스트
            visited = set()
            for i in range(args.max_inst_num):
                if valid_mask[i] == 0 or  i in visited:
                    continue
                group = [i]
                visited.add(i)
                for j in range(i + 1, args.max_inst_num):
                    if valid_mask[j] == 0 or j in visited:
                        continue
                    iou = compute_3d_iou(instance_segids[i], instance_segids[j])
                    if iou >= args.overlap_iou_thres:
                        group.append(j)
                        visited.add(j)
                groups.append(group)
            valid_mask = [0] * args.max_inst_num
            
            if split == 'train' and scannet_attrs:
                gt_segids = scannet_attrs[scan_id]["segids"]
                for group in groups:
                    best_pred, best_iou = None, -1.0
                    for pid in group:
                        for gt_seg in gt_segids:
                            iou = compute_3d_iou(instance_segids[pid], gt_seg)
                            if iou > best_iou:
                                best_iou = iou
                                best_pred = pid
                    if best_pred is not None and best_iou > 0.0:
                        print(best_iou)
                        valid_mask[best_pred] = 1
                    else:
                        valid_mask[group[0]] = 1
            else:
                for group in groups:
                    valid_mask[group[0]] = 1
            
            group_infos[scan_id] = groups
            total = args.max_inst_num
            unique = len(groups)
            
            if args.islayout:
                max_thickness = 0.5
                inst_np = inst_locs.numpy()
                for i in range(len(inst_np)):
                    if valid_mask[i] == 0:
                        continue
                    # dx, dy, dz
                    w, h, d = inst_np[i, 3], inst_np[i, 4], inst_np[i, 5]
                    thickness = min(w,h,d)
                    print(thickness)
                    if thickness > max_thickness:
                        # 너무 두꺼우면 invalid
                        valid_mask[i] = 0
            
            print(scan_id, valid_mask, 1 - (unique/total))
            print(scan_id, instance_class_labels)
            scans[scan_id] = {
                'objects': instance_class_labels,  # (n_obj,)
                'locs': inst_locs,                   # (n_obj, 6) center xyz, whl,
                'valid': valid_mask
            }
        else:
            scans[scan_id] = {
                'objects': instance_class_labels,  # (n_obj, )
                'locs': inst_locs,  # (n_obj, 6) center xyz, whl,
            }

    torch.save(scans, os.path.join(output_dir, f"scannet_{args.segmentor}_{split}_attributes{args.version}_{args.overlap_iou_thres}.pt"))
    # 그룹 정보도 따로 저장
    if '_nonoverlap' in args.version:
        torch.save(group_infos, os.path.join(output_dir, f"scannet_{args.segmentor}_{split}_groups{args.version}_{args.overlap_iou_thres}.pt"))
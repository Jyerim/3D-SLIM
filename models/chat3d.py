import random
import logging
from abc import ABC

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from .modeling_qwen2 import Qwen2ForCausalLM
from .modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from models.position_embedding import PositionEmbeddingCoordsSine
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from utils.pc_util import shift_scale_points
import contextlib
from dataset.base_dataset import update_caption, recover_caption
from models.utils import visualize_attn_maps, _plot_heatmap
import math
logger = logging.getLogger(__name__)

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()


def print_grad_status(model):
    """Call this function after losses.backward()
    and it will find out all variables without grad, which
    means that the varaible is not in the graph.
    """
    for name, p in model.named_parameters():
        print('{:80s}{:20s}{:20s}{}'.format(name,
            '(Trainable)' if p.requires_grad else '(Fixed)',
            '(Has grad):' if p.grad is not None else '(No grad backward):',
            list(p.shape)))


class Chat3D(nn.Module):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        llama_model_path = config.model.llama_model_path
        self.low_resource = config.model.low_resource
        self.max_txt_len = config.model.max_txt_len
        self.end_sym = config.model.end_sym
        self.system_path = config.model.system_path
        self.instruction_path = config.model.instruction_path
        self.role = config.model.role
        self.input_dim = config.model.input_dim
        self.img_input_dim = config.model.img_input_dim
        self.max_obj_num = config.model.max_obj_num
        self.bidirection = config.model.bidirection
        self.mask_type = config.model.mask_type
        self.positional_emb = config.model.positional_emb
        self.top_k_min = config.model.top_k_min
        self.top_k_max = config.model.top_k_max
        
        self.debug = config.debug
        if not self.debug:
            logger.info('Loading LLM')
            if "vicuna" in llama_model_path:
                self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False, legacy=False)
            else:
                self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False )
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            _path = llama_model_path.lower()
            if "vicuna" in _path:
                _model_cls = LlamaForCausalLM
                _pad_token = False
            elif ("llama-3" in _path) or ("llama3" in _path):
                _model_cls = LlamaForCausalLM
                _pad_token = True
            elif "qwen2" in _path:
                _model_cls = Qwen2ForCausalLM
                _pad_token = True
            else:
                raise NotImplementedError(f"Unsupported llm_model_path: {llama_model_path}")

            if self.low_resource:
                self.llama_model = _model_cls.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    attn_implementation="eager"
                )
            else:
                self.llama_model = _model_cls.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager"
                )
            if _pad_token:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

            logger.info("freeze LLM")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
                    
            if config.model.use_lora:
                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in [
                                        "instance2embed",
                                        "hidden_state2query"
                                    ]
                                ]
                            )
                            and any([x in name for x in lora_target_modules])
                        ):
                            lora_module_names.add(name)
                    return sorted(list(lora_module_names))
            
                lora_target_modules = find_linear_layers(self.llama_model, config.lora.lora_target_modules)

                lora_config = LoraConfig(
                    r=config.lora.lora_r,
                    lora_alpha=config.lora.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=config.lora.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                self.llama_model.print_trainable_parameters()
                self.llama_model.model.lm_head.weight.requires_grad = True
                self.llama_model.model.lm_head.weight.data = self.llama_model.model.lm_head.weight.data.float()
                self.llama_model.print_trainable_parameters()
                self.llama_model.model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.model.embed_tokens.weight.data = self.llama_model.model.model.embed_tokens.weight.data.float()
                self.llama_model.print_trainable_parameters()
                
            else:
                self.llama_model.lm_head.weight.requires_grad = True
                self.llama_model.lm_head.weight.data = self.llama_model.lm_head.weight.data.float()
                self.llama_model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.embed_tokens.weight.data = self.llama_model.model.embed_tokens.weight.data.float()
            
            self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            objid_tokens = []
            for i in range(self.max_obj_num):
                objid_tokens.append(f"<OBJ{i:03}>")
            self.objid_start_idx = self.ori_vocab_size = len(self.llama_tokenizer)
            self.llama_tokenizer.add_tokens(objid_tokens, special_tokens=True)
            self.objid_end_idx = len(self.llama_tokenizer)
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            self.llm_dim = self.llama_model.config.hidden_size
            
            logger.info('Loading LLM Done')
        else:
            self.llama_model = None
            self.llm_dim = 4096
            
        for name, param in self.llama_model.named_parameters():
            if 'spatial_bias' in name:
                param.requires_grad = True
        
        self.object_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim),
        )
        self.object_img_proj = nn.Sequential(
            nn.Linear(self.img_input_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim),
        )
        
        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        with open(self.instruction_path, "r") as f:
            self.instruction = "\n".join([x.strip() for x in f.readlines()])

        if not self.debug:
            self.p_0_embed, self.p_1_embed = self.prepare_fixed_embed()
        self.last_embed = None


    def get_objid_embeds(self):
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx] # max_obj_num * 4096
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        return objid_embeds
    
    def llama_embed_tokens(self, token_ids):
        if self.config.model.use_lora:
            return self.llama_model.model.model.embed_tokens(token_ids)
        else:
            return self.llama_model.model.embed_tokens(token_ids)

    def prepare_fixed_embed(self):
        prompt = self.system + " " + self.instruction + " " + self.role[0] + ": " 
        p_0, p_1 = prompt.split("<REPLACE>")
        p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=True)
        p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False)
        p_0_embed = self.llama_embed_tokens(p_0_token.input_ids).squeeze(0).detach()
        p_1_embed = self.llama_embed_tokens(p_1_token.input_ids).squeeze(0).detach()
        return p_0_embed, p_1_embed


    def get_text_emb(self, text, device="cpu"):
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        embeds = self.llama_embed_tokens(text_tokens.input_ids)
        indices = text_tokens.input_ids >= self.ori_vocab_size
        indices = (indices * 1).unsqueeze(-1)
        embeds = (1 - indices) * embeds.detach() + indices * embeds

        return embeds

    def encode_object_feat(self, feat, img_feat):
        feat = torch.nan_to_num(torch.nn.functional.normalize(feat, dim=-1, eps=1e-6))
        img_feat = torch.nan_to_num(torch.nn.functional.normalize(img_feat, dim=-1, eps=1e-6))
        return feat, img_feat
    
    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        # pos (bs, obj_num, 3)
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn

    def get_min_max_coord(self, xyz, scene_mask):
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)
        masked_xyz_min = torch.where(scene_mask, xyz, torch.full_like(xyz, float('inf')))
        masked_xyz_max = torch.where(scene_mask, xyz, torch.full_like(xyz, float('-inf')))
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs
    
    def get_object_list_embed(self, embed_obj, embed_img, scene_mask, scene_locs):
        valid_ids = torch.where(scene_mask)[0].tolist()
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx] # max_obj_num * 4096
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        
        valid_ids = torch.where(scene_mask)[0].tolist()
        valid_embeds = embed_obj[valid_ids]
        valid_img_embeds = embed_img[valid_ids]
        valid_center_locs = scene_locs[valid_ids] # (N, 3)

        proj_valid_embed = self.object_proj(valid_embeds)
        proj_valid_img_embed = self.object_img_proj(valid_img_embeds)
        
        selected_objid_embeds = objid_embeds[valid_ids]
        object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
        object_list_embed[0::3, :] = selected_objid_embeds
        object_list_embed[1::3, :] = proj_valid_embed
        object_list_embed[2::3, :] = proj_valid_img_embed
        return object_list_embed, proj_valid_embed, proj_valid_img_embed, valid_center_locs
    
    def get_min_max_coord(self, xyz, scene_mask):
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)
        masked_xyz_min = torch.where(scene_mask, xyz, torch.full_like(xyz, float('inf')))
        masked_xyz_max = torch.where(scene_mask, xyz, torch.full_like(xyz, float('-inf')))
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs


    def make_density_adaptive_mask(
        self,
        obj_centers: torch.Tensor,
        Lobj: int,
        chunk: int = 3,
        k_min: int = 2,
        k_max: int = 20,
        dtype=None,
        device=None,
    ) -> torch.Tensor:
        """
        obj_centers: [N, 3] tensor (각 object center 좌표)
        Lobj: 전체 object 토큰 개수 (3N)
        chunk: object 당 토큰 수
        k_min, k_max: 최소/최대 neighbor 개수
        return: [Lobj, Lobj] block mask
        """
        N = Lobj // chunk
        if N == 0:
            return torch.zeros((Lobj, Lobj), dtype=dtype, device=device)

        # 거리 행렬 [N, N]
        diff = obj_centers[:, None, :] - obj_centers[None, :, :]
        dist = torch.norm(diff, dim=-1)
        dist.fill_diagonal_(0.0)

        # object별 평균 거리 [N]
        mean_dist = math.sqrt(3) - (dist.sum(dim=-1) / (N - 1))

        # normalize mean_dist → [0,1]
        min_d, max_d = mean_dist.min(), mean_dist.max()
        norm_density = (mean_dist - min_d) / (max_d - min_d + 1e-6)

        # density → adaptive k (sparse=작은 k, dense=큰 k)
        k_vals = (k_max - k_min) * norm_density + k_min
        k_vals = k_vals.round().long()  # 각 object별 k_i
        k_vals = k_vals.clamp(min=k_min, max=min(k_max, N-1))
        
        # block mask 초기화
        blk = torch.zeros((Lobj, Lobj), dtype=dtype, device=device)

        # neighbor 선택
        dist.fill_diagonal_(float('inf'))
        for i in range(N):
            k_i = k_vals[i].item()
            if k_i >= N:  # neighbor가 부족하면 전체 허용
                neighbor_ids = list(range(N))
            else:
                _, nn_idx = torch.topk(dist[i], k=k_i, largest=False)
                neighbor_ids = nn_idx.tolist()

            neighbor_ids = [i] + neighbor_ids  # 자기 자신 포함
            for nb in neighbor_ids:
                obj_range = slice(i * chunk, (i + 1) * chunk)
                nb_range = slice(nb * chunk, (nb + 1) * chunk)
                blk[obj_range, nb_range] = 1.0

        return blk

    def forward_train(self, scene_feat, scene_img_feat, scene_locs, scene_mask, \
        assigned_ids, questions, answers, is_eval=False, **kwargs):
        
        
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat)

        device = object_embed.device
        batch_size = object_embed.shape[0]
        
        mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
        norm_locs = shift_scale_points(scene_locs[:, :, :3], src_range=[mins, maxs])
            
        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        p_0_embed = self.p_0_embed.to(device)
        p_1_embed = self.p_1_embed.to(device)
        
        object_list_intervals = []
        center_locs_list = []
        boundaries = []
        for i, question in enumerate(questions):
            prompt = f"{question} {self.role[1]}: "
            prompt_embed = self.get_text_emb(prompt, device=device).squeeze(0) # (Lq, D)
            
            valid_center_locs = None

            object_list_embed, proj_valid_obj, proj_valid_img, valid_center_locs = self.get_object_list_embed(
                object_embed[i],
                object_img_embed[i],
                scene_mask[i],
                norm_locs[i]
            )
            wrapped_embed = torch.cat([
                p_0_embed,
                object_list_embed,
                p_1_embed,
                prompt_embed
            ], dim=0)
            L0 = p_0_embed.size(0)
            Lobj = object_list_embed.size(0)
            L1 = p_1_embed.size(0)
            Ltxt = prompt_embed.size(0)
            boundary = torch.tensor([L0, Lobj, L1, Ltxt], dtype=torch.long, device=device)
            st = L0
            ed = st + Lobj
            boundaries.append(boundary)
            
            object_list_intervals.append((st, ed))
            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(wrapped_embed.device)

            empty_target = (
                torch.ones(wrapped_attn.shape[0], dtype=torch.long).to(device).fill_(-100)
            )

            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(device)

            answer_target = to_regress_token.input_ids.masked_fill(
                to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100
            ).squeeze(0)

            to_regress_embed = self.get_text_emb(answer, device=device).squeeze(0)

            target = torch.cat([empty_target, answer_target], dim=0)
            input_embed = torch.cat([wrapped_embed, to_regress_embed], dim=0)
            attn = torch.cat([wrapped_attn, to_regress_token.attention_mask[0]], dim=0)
            
            input_embed_list.append(input_embed)
            center_locs_list.append(valid_center_locs)
            attn_list.append(attn)
            target_list.append(target)
            max_seq_len = max(max_seq_len, target.shape[0])
        
        max_seq_len = min(2048, max_seq_len)

        def pad_and_trim(tensor_list, max_len, batch_first=True, padding_value=0):
            padded = pad_sequence(tensor_list, batch_first=batch_first, padding_value=padding_value)
            if padded.shape[1] > max_len:
                return padded[:, :max_len]
            return padded
        
        boundaries = torch.stack(boundaries, dim=0)
        input_embeds = pad_and_trim(input_embed_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        if center_locs_list[0] is not None:
            center_locs = pad_and_trim(center_locs_list, max_seq_len, batch_first=True, padding_value=-100).to(device)
        targets = pad_and_trim(target_list, max_seq_len, batch_first=True, padding_value=-100).to(device)
        attention_mask = pad_and_trim(attn_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        if self.bidirection:
            input_dtype = input_embeds.dtype
            causal_mask = torch.ones((max_seq_len, max_seq_len), dtype=input_dtype, device=device)
            causal_mask = torch.tril(causal_mask, diagonal=0)
            causal_mask = causal_mask[None, None, :, :].expand(input_embeds.shape[0], 1, -1, -1).clone()
            padding_mask = causal_mask[..., :].eq(1.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :] = causal_mask[..., :].masked_fill(padding_mask, 0.0)
            CHUNK=3
            for i in range(causal_mask.shape[0]):

                L0, Lobj, L1, Ltxt = boundaries[i]
                vis1_s, vis1_e = L0, L0 + Lobj
                sys2_s, sys2_e = vis1_e, vis1_e + L1
                txt_s,  txt_e  = sys2_e, sys2_e + Ltxt
                 
                if self.mask_type == 'GeoMask+InstMask':
                    if Lobj > 0:
                        blk = self.make_density_adaptive_mask(
                            obj_centers=center_locs_list[i],
                            Lobj=Lobj,
                            chunk=CHUNK,
                            k_min=self.top_k_min,
                            k_max=self.top_k_max,
                            dtype=causal_mask.dtype,
                            device=causal_mask.device
                        )
                        causal_mask[i, :, vis1_s:vis1_e, vis1_s:vis1_e] = blk   
                    causal_mask[i, :, vis1_s:vis1_e, txt_s:txt_e] = 1.0 # object - text                                 
            attention_mask = causal_mask

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
                
        return dict(
            loss=outputs.loss,
            obj_norm=proj_valid_obj.norm(dim=-1).mean().detach().cpu(),
            obj_img_norm=proj_valid_img.norm(dim=-1).mean().detach().cpu(),
            objid_norm=self.get_objid_embeds().norm(dim=-1).mean().detach().cpu(),
            max_seq_len=max_seq_len
        )

    def evaluate(self, scene_feat, scene_img_feat, scene_locs, scene_mask, assigned_ids, \
        obj_ids, custom_prompt, is_eval=False, **kwargs):
        
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat)

        device = object_embed.device
        batch_size, obj_num = object_embed.shape[:2]
        
        mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
        norm_locs = shift_scale_points(scene_locs[:, :, :3], src_range=[mins, maxs])
            
        output_texts = []
        p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
        p_1_embed = self.p_1_embed.to(device).unsqueeze(0)
        
        for i in range(batch_size):
            tmp_prompt = f" {custom_prompt[i]} {self.role[1]}: "
            tmp_prompt = update_caption(tmp_prompt, assigned_ids[i])
            prompt_embed = self.get_text_emb(tmp_prompt, device=device)

            object_list_embed, _, _, valid_center_locs = self.get_object_list_embed(
                object_embed[i],
                object_img_embed[i],
                scene_mask[i],
                norm_locs[i]
            )
            valid_center_locs = valid_center_locs.unsqueeze(0)
            object_list_embed = object_list_embed.unsqueeze(0)
            
            wrapped_embed = torch.cat([
                p_0_embed,
                object_list_embed,
                p_1_embed,
                prompt_embed
            ], dim=1)
            L0 = p_0_embed.shape[1]
            Lobj = object_list_embed.shape[1]
            L1 = p_1_embed.shape[1]
            Ltxt = prompt_embed.shape[1]
            boundary = (L0, Lobj, L1, Ltxt)
            st = L0
            ed = st + Lobj

            self.llama_tokenizer.padding_side = "left"
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            seq_len = wrapped_embed.size(1)
            # 기본 causal mask (1=keep, 0=mask)
            attention_mask = torch.ones((seq_len, seq_len), dtype=wrapped_embed.dtype, device=device)
            attention_mask = torch.tril(attention_mask, diagonal=0)
            attention_mask = attention_mask[None, None, :, :].expand(1, 1, -1, -1).clone()
            
            if self.bidirection:

                CHUNK = 3  # 3개 feature가 1개 chunk

                vis1_s, vis1_e = L0, L0 + Lobj
                sys2_s, sys2_e = vis1_e, vis1_e + L1
                txt_s,  txt_e  = sys2_e, sys2_e + Ltxt
                
                if self.mask_type == 'GeoMask+InstMask':
                    if Lobj > 0:
                        blk = self.make_density_adaptive_mask(
                            obj_centers=valid_center_locs.squeeze(0),
                            Lobj=Lobj,
                            chunk=CHUNK,
                            k_min=self.top_k_min,
                            k_max=self.top_k_max,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device
                        )
                        attention_mask[:, :, vis1_s:vis1_e, vis1_s:vis1_e] = blk       
                    attention_mask[:, :, vis1_s:vis1_e, txt_s:txt_e] = 1.0 # object - text

            stop_words_ids = [torch.tensor([ 14711]).to(wrapped_embed.device), torch.tensor([198,  14711]).to(wrapped_embed.device), torch.tensor([82, 29]).to(wrapped_embed.device), torch.tensor([524]).to(wrapped_embed.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

            with self.maybe_autocast():
                try:
                    outputs = self.llama_model.generate(
                        inputs_embeds=wrapped_embed,
                        max_new_tokens=self.max_txt_len,
                        stopping_criteria=stopping_criteria,
                        num_beams=5,
                        min_length=1,
                        repetition_penalty=3.0,
                        length_penalty=1,
                        temperature=1.0,
                        customized_mask=attention_mask,
                        pad_token_id=self.llama_tokenizer.eos_token_id,
                    )
                except RuntimeError as e:
                    with open(f"error_prompt_{i}.txt", "w") as f:
                        f.write(tmp_prompt)
                    raise e
            output_token = outputs[0]

            del outputs
            torch.cuda.empty_cache()
            output_text = self.llama_tokenizer.decode(output_token)
            output_text = output_text.split(self.end_sym)[0]
            output_text = output_text.replace('  ', ' ').replace(' .', '.').strip()
            output_text = recover_caption(output_text, assigned_ids[i].tolist())
            output_texts.append(output_text)
        return output_texts


    def forward(self, **kwargs):
        if "answers" in kwargs:
            return self.forward_train(**kwargs)
        if "custom_prompt" in kwargs:
            return self.evaluate(**kwargs)
        return None

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
    

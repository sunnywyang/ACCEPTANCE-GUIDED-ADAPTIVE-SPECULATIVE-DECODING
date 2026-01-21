# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.

import copy
import torch.nn as nn
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import time
from transformers.activations import ACT2FN
from abc import abstractmethod, ABC

try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor

"""
mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The mLSTM extends the traditional LSTM by using a matrix memory state and exponential gating,
allowing for enhanced storage capacities and improved performance on long-range dependencies.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
#sys.path.append('/root/autodl-tmp/EAGLE/xLSTM/xlstm/blocks/mlstm')
#from layer import mLSTMLayer,mLSTMLayerConfig
import torch
from torch import nn
import sys
'''sys.path.append('/root/autodl-tmp/EAGLE/xLSTM')
from xlstm.blocks.slstm.layer import sLSTMLayer
from xlstm.blocks.mlstm.layer import mLSTMLayer,mLSTMLayerConfig
from xlstm.components.feedforward import create_feedforward
import xlstm.components.ln as ln
from xlstm.components.feedforward import FeedForwardConfig'''

'''import torch
import numpy as np
sys.path.append('/root/autodl-tmp/xlstm-main')
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge'''


'''class NormLayer(nn.Module, ABC):
    """Base class for normalization layers.
    This class contains optional learnable weight and bias parameters.
    
    Args:
        num_features: The number of features in the input tensor.
        eps: A small value to avoid division by zero.
        use_weight: Whether to use a learnable weight.
        use_bias: Whether to use a learnable bias.
        force_float32_reductions: Whether to force float32 reductions.   
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.force_float32_reductions = force_float32_reductions

        if use_weight:
            self.weight = nn.Parameter(torch.ones(num_features))
        else:
            self.weight = None

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.bias = None

    def _apply_weight_bias(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError'''

'''class RMSNorm(NormLayer):
    """Root mean square normalization layer implementation similar
    to https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html. 

    It normalizes the input tensor by the root mean square of the last dimension.

    Args:
        num_features: The number of features in the input tensor.
        eps: A small value to avoid division by zero.
        use_weight: Whether to use a learnable weight.
        use_bias: Whether to use a learnable bias.
        force_float32_reductions: Whether to force float32 reductions.
    """

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        # apply rms norm over the last dimension, i.e. D dimension
        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(in_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        x = self._rms_normalize(x)
        x = self._apply_weight_bias(x)
        return x'''




'''class xLSTM(nn.Module):
  def __init__(self, layers, scfg=None, mcfg=None, fcfg=None):
    super().__init__()
    self.layers = layers
    embedding_dim = (mcfg.embedding_dim if mcfg is not None else scfg.embedding_dim)
    self.xlstm_norm = nn.ModuleList()
    self.xlstm_blocks = nn.ModuleList()
    self.ffn_norm = nn.ModuleList()
    self.ffn = nn.ModuleList()
    if scfg is not None:
      scfg.__post_init__()
    if mcfg is not None:
      mcfg.__post_init__()
    if fcfg is not None:
      fcfg.__post_init__()
    for i in range(len(layers)):
      self.xlstm_norm.append(ln.LayerNorm(ndim=embedding_dim, weight=True, bias=False))
      if layers[i] == 's':
        self.xlstm_blocks.append(sLSTMLayer(scfg))
      else:
        self.xlstm_blocks.append(mLSTMLayer(mcfg))
      self.ffn_norm.append(ln.LayerNorm(ndim=embedding_dim, weight=True, bias=False))
      self.ffn.append(create_feedforward(fcfg))
    self.post_blocks_norm = ln.LayerNorm(ndim=embedding_dim)
    self.reset_parameters()

  def forward(self, x, hidden):
    if hidden is None:
      hidden = {}
    for block_idx, block in enumerate(self.xlstm_blocks):
      if self.layers[block_idx] == 's':
        x, hidden[f'block_{block_idx}'] = block(self.xlstm_norm[block_idx](x), hidden.get(f'block_{block_idx}', None), return_last_state=True)
      else:
        x = block(self.xlstm_norm[block_idx](x))
      x = x + self.ffn[block_idx](self.ffn_norm[block_idx](x))
    x = self.post_blocks_norm(x)
    return x, hidden

  def reset_parameters(self):
    for i in range(len(self.layers)):
      self.xlstm_norm[i].reset_parameters()
      self.xlstm_blocks[i].reset_parameters()
      self.ffn_norm[i].reset_parameters()
      self.ffn[i].reset_parameters()
    self.post_blocks_norm.reset_parameters()'''
class DeepseekMLP(nn.Module):
    def __init__(self, config, hidden_size = None, intermediate_size = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([DeepseekMLP(config, intermediate_size = config.moe_intermediate_size) for i in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(config=config, intermediate_size = intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
from types import SimpleNamespace



def len_list(x, n):
    return [i for i in x if len(i) <= n]

#draft model
class Model(nn.Module):
    #config: 模型配置,load_emb: 是否加载预训练的嵌入,这里的path是basemodel的path,top_k==8可以参考EAGLE-2的rerank情况
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=-15):
        super().__init__()

        #self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    #model的embedding层路径
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor # 将加载的权重赋值给embedding层

        self.act = ACT2FN[config.hidden_act]
        self.hidden_size=config.hidden_size

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = threshold

        #======MOE设置的参数======
        '''
        self.n_routed_experts=4,       # 路由4个专家
        self.moe_intermediate_size=config.intermediate_size
        self.n_shared_experts=None    # 如果有共享专家，可设置为一个正整数，否则为None
        self.scoring_func='softmax'   # 门控使用softmax计算得分
        self.aux_loss_alpha=0.1       # 辅助损失超参数
        self.seq_aux=False           # 是否进行序列级的辅助损失计算
        self.norm_topk_prob=True       # 是否对前k专家概率进行归一化'''
        self.down_proj=nn.Linear(config.hidden_size * 2, config.hidden_size)
        config_moe = SimpleNamespace(
        
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        pretraining_tp=1,         # 如不需要并行切片，可设为1
        hidden_act="silu",        # 使用gelu激活函数
        num_experts_per_tok=6,    # 每个 token 分配2个专家
        n_routed_experts=28,       # 路由4个专家
        moe_intermediate_size=1408,
        n_shared_experts=None,    # 如果有共享专家，可设置为一个正整数，否则为None
        scoring_func='softmax',   # 门控使用softmax计算得分
        aux_loss_alpha=0.1,       # 辅助损失超参数
        seq_aux=False,            # 是否进行序列级的辅助损失计算
        norm_topk_prob=False       # 是否对前k专家概率进行归一化
        )



        self.moe=DeepseekMoE(config_moe)

        
        #==============现在使用的mlstmlayer============
        '''self.mlstm_layer = mLSTMLayer(mLSTMLayerConfig(
            embedding_dim=config.hidden_size, 
            qkv_proj_blocksize=32,
            conv1d_kernel_size =8 ,
            num_heads=8, 
            proj_factor=2,
            dropout=0.1
            
            
        ))

        self.out_norm = RMSNorm(
                num_features=config.hidden_size,
                eps=1e-6,
                use_weight=True,
                use_bias=False,
                force_float32_reductions=False,
            )
        self.in_norm = RMSNorm(
                num_features=config.hidden_size,
                eps=1e-6,
                use_weight=True,
                use_bias=False,
                force_float32_reductions=False,
            )

        self.mlstm_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)'''

        '''mlstm_config = xLSTMLargeConfig(
            embedding_dim=config.hidden_size,
            num_heads=4,            # 根据需要设置头数
            num_blocks=1,           # 单层 mLSTMBlock
            vocab_size=config.vocab_size,  # 虽然内部的 embedding 不会用到，但此处需要提供
            use_bias=bias,
            norm_eps=1e-6,
            norm_reduction_force_float32=True,
            add_out_norm=False,     # 这里可以不加最后的归一化
            # 其它参数按默认或根据需求设置
        )
        # 只使用 backbone 部分（即 xLSTMLargeBlockStack）来提取特征
        self.mlstm_backbone = xLSTMLargeBlockStack(mlstm_config)'''

        self.logsoftmax = nn.LogSoftmax(dim=-1)
        # embedding层frozen
        for param in self.embed_tokens.parameters():
            param.requires_grad = False


        #self.se_layer=SELayer(config.hidden_size)


        #self.skip_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
       #========现在在使用的================
        '''self.gate_layer = nn.Linear(config.hidden_size*2 , config.hidden_size )
        self.gate_proj = nn.Linear(config.hidden_size*2 , config.hidden_size * 3)
        self.value_proj = nn.Linear(config.hidden_size*2 , config.hidden_size * 3)
        self.down_proj = nn.Linear(config.hidden_size * 3, config.hidden_size)'''
        '''#========现在在使用的================
        self.gate_proj = nn.Linear(config.hidden_size*3 , config.hidden_size * 4)
        self.value_proj = nn.Linear(config.hidden_size*3 , config.hidden_size * 4)
        self.down_proj = nn.Linear(config.hidden_size * 4, config.hidden_size)'''

        '''self.gate_layer = nn.Linear(config.hidden_size*2 , config.hidden_size )
        self.gate_proj = nn.Linear(config.hidden_size*2 , config.hidden_size * 3)
        self.value_proj = nn.Linear(config.hidden_size*2 , config.hidden_size * 3)
        self.down_proj = nn.Linear(config.hidden_size * 3, config.hidden_size)'''



        #self.skip_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        '''layers=[]
        for i in range(1):
            layers.append(nn.Sequential(
                nn.Linear(config.hidden_size*2 , config.intermediate_size ),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
           # nn.GELU(),
            #nn.Linear(config.hidden_size *2 , config.hidden_size)

        ))
        self.layer = nn.Sequential(*layers)'''
        '''self.gate_proj = nn.Linear(config.hidden_size * 2, config.hidden_size * 4)
        self.value_proj = nn.Linear(config.hidden_size * 2, config.hidden_size * 4)
        self.down_proj = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.gate_layer=nn.Linear(config.hidden_size * 2, config.hidden_size )'''
        


        # GLU分支
        '''self.gate_proj = nn.Linear(config.hidden_size * 2, config.hidden_size * 4)
        self.value_proj = nn.Linear(config.hidden_size * 2, config.hidden_size * 4)
        self.down_proj = nn.Linear(config.hidden_size * 4, config.hidden_size)
         # 残差分支用于保持重要信息
        #self.skip_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.logsoftmax = nn.LogSoftmax(dim=-1)
        # embedding层frozen
        for param in self.embed_tokens.parameters():
            param.requires_grad = False'''

    '''def reset_parameters(self):
        self.mlstm_layer.reset_parameters()'''

    def init_tree(self):
        #创建了一个self.top_k x self.top_k的单位矩阵，其中对角线上的元素为1，其余元素为0。
        # 这个矩阵用于表示树状结构中每个节点的初始状态，其中self.top_k表示每个节点可以扩展的最大子节点数。
        #None索引扩展了矩阵的维度，使其成为一个四维张量，以适应后续的广播操作。
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        #创建了一个大小为self.top_k的零向量，被初始化为一个一维张量，表示树状结构中每个节点的初始位置ID。用于跟踪树状结构中每个节点的位置。
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        #确保树掩码被移动到与词嵌入权重相同的设备上（如GPU），以便于后续的计算。
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None
    
       # forward 支持两种模式：
    # 1. 普通前向传播（use_step=False），适用于训练或一次性处理整个序列
    # 2. 生成任务（use_step=True），逐步生成时调用 mlstm_layer.step，并返回新的状态
    def forward(
            self,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            #return_laststate:bool=True
            #use_step: bool = False,
            
    ) -> torch.Tensor:
        """
        hidden_states: [B, seq_len, hidden_size] 先前的 hidden state
        input_ids: [B, seq_len] 输入 token id
        use_step: 是否使用 mlstm_layer 的 step 方法进行逐步生成（仅适用于 seq_len=1的情况）
        mlstm_state, conv_state: 若 use_step=True，需要传入上一步的状态
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        device = hidden_states.device

        # 获取输入 token 的 embedding（注意：embedding 层被冻结，故放在 no_grad 下）
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        # 拼接输入 embedding 和 hidden state，维度为 [B, seq_len, hidden_size*2]
        combined = torch.cat((inputs_embeds, hidden_states), dim=-1)
        # 通过 mlstm_proj 将拼接后的特征映射到 hidden_size 维度
        output=self.moe(self.down_proj(combined))
        return output
        '''out,state= self.xlstmlarge(mlstm_input)

        return out'''
        
        
        

    
    
            #new_states = None

        # 残差连接
        #new_hidden_states = mlstm_output + hidden_states

        # 如果是生成任务，返回新的状态以便下一步调用
        #if use_step:
        #    return new_hidden_states, new_states
        #else:
        #    return new_hidden_states

        '''#生成单个token的hidden_state
    def forward(
            self,
            hidden_states,
            input_ids,
            position_ids: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            std=None
    ):

        #seq_length=1
        #print(hidden_states.shape)  # 在问题解包之前添加这行代码

        batch_size, seq_length, hidden_size = hidden_states.shape
        device = hidden_states.device
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        combined = torch.cat((inputs_embeds, hidden_states), dim=-1)
        
        # 维度适配（如果需要）
        mlstm_input = self.mlstm_proj(combined)
        
        
        # 通过mLSTM进行特征提取
        mlstm_output = self.mlstm_layer(self.out_norm(mlstm_input))
        
        # 残差连接保持稳定性
        new_hidden_states = mlstm_output + hidden_states
        
        # ================= 后续处理保持不变 =================
        # （可以在此处添加其他处理层）
        
        return new_hidden_states  # 确保输出维度为'''
    

    



        


        '''if pre_hidden_states.shape[1]==0:
            pre_hidden_states=torch.zeros(batch_size, seq_length, hidden_size,device=device)
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        
        
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        pre_hidden_states=pre_hidden_states.to(hidden_states.dtype)
        combined_features=torch.cat((inputs_embeds,pre_hidden_states, hidden_states), dim=-1)
        gate =self.act(self.gate_proj(combined_features))
        value =self.value_proj(combined_features)
        new_hidden_states =gate * value 
# GLU门控
        new_hidden_states=self.down_proj(new_hidden_states)
        return new_hidden_states'''

        '''hidden_states= torch.cat((inputs_embeds,hidden_states),dim=-1)
        return self.layer(hidden_states)'''
        #拼接输入和隐藏状态并通过全连接层

        # 随机 Mask 操作
        #mask = torch.rand(inputs_embeds.shape[:2]) > 0.05  # 15% 进行 Mask
        #mask = mask.unsqueeze(-1).to(inputs_embeds.device)
        #masked_inputs_embeds = inputs_embeds * mask
        #masked_hidden_states = hidden_states * mask
# 门控特征融合
        #new_gate=torch.sigmoid(inputs_embeds)
        #hidden_states= new_gate * pre_hidden_states + (1 - new_gate) * hidden_states
        #===========目前使用的是这个===============
        '''combined_features = torch.cat((inputs_embeds, hidden_states), dim=-1)
        gate = torch.sigmoid(self.gate_layer(combined_features))
        fused_features = gate * inputs_embeds + (1 - gate) * hidden_states
        gate =self.act(self.gate_proj(combined_features))
        value =self.value_proj(combined_features)
        new_hidden_states =gate * value 
# GLU门控
        new_hidden_states=self.down_proj(new_hidden_states)
        #gate_p=torch.sigmoid(self.gate_layer(combined_features))
        #res=gate_p*self.gate_layer(combined_features)
        hidden_states=new_hidden_states+fused_features
        #print(hidden_states.shape)


        return hidden_states'''



        '''combined_features= torch.cat((inputs_embeds,hidden_states),dim=-1)
#hidden states=self.layer(hidden states)
#将x和y的特征进行外积
        gate = torch.sigmoid(self.gate_layer(combined_features))# [batch size, seq lel.dden size]
        fused_features = gate * inputs_embeds +(1 - gate)* hidden_states # 加权融合
        gate =self.act(self.gate_proj(combined_features))
        value =self.value_proj(combined_features)
        new_hidden_states =gate * value 
# GLU门控
        new_hidden_states=self.down_proj(new_hidden_states)
        hidden_states=new_hidden_states+fused_features
        return hidden_states'''
        '''if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()


        position_ids = position_ids.to(hidden_states.device)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        inputs_embeds = inputs_embeds + self.pe[:, position_ids]'''
        
        # 假设 input_embedding 和 prev_hidden_state 具有相同的形状 [batch_size, seq_len, hidden_size]
# 使用门控机制来融合这两个特征
        #combined_features=torch.cat((inputs_embeds, hidden_states), dim=-1)


        #inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        #！！！！！！！！！特征和embedding的融合！！
        # 拼接输入和隐藏状态并通过全连接层
        #seq_length=1

        #position_embeds = self.position_encoding(position_ids=position_ids, d_model=self.hidden_size, device=device)
    # 注意裁剪到 seq_length 的范围
       # inputs_embeds = inputs_embeds + position_embeds
        
        # 假设 input_embedding 和 prev_hidden_state 具有相同的形状 [batch_size, seq_len, hidden_size]
# 使用门控机制来融合这两个特征


        #！！！！！！！！！特征和embedding的融合！！
        # 拼接输入和隐藏状态并通过全连接层


        #device = inputs_embeds.device
        #position_embeds = self.position_encoding(position_ids=position_ids, d_model=self.hidden_size, device=device)
    # 注意裁剪到 seq_length 的范围
       # inputs_embeds = inputs_embeds + position_embeds
       # inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        # 假设 input_embedding 和 prev_hidden_state 具有相同的形状 [batch_size, seq_len, hidden_size]
# 使用门控机制来融合这两个特征
        '''inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        hidden_states=torch.cat((inputs_embeds, hidden_states), dim=-1)
                # 合并残差
        hidden_states = self.layer(hidden_states)
        #hidden_states=hidden_states+residual
        return hidden_states'''

        '''new_hidden_states=torch.cat((inputs_embeds, hidden_states), dim=-1)
        #残差链接
        residual = self.skip_proj(new_hidden_states) 
 # 主干路径
        gate = torch.sigmoid(self.gate_layer(hidden_states))
        combined_hidden = gate * inputs_embeds + (1 - gate) * hidden_states
        gate=self.act(self.gate_proj(combined_hidden))
        value = self.value_proj(combined_hidden)
        hidden_states = gate * value  # GLU门控
        hidden_states = self.down_proj(hidden_states)
        hidden_states=hidden_states+residual
        return hidden_states'''
        #!!!!!!!!!特征和embedding的融合!!







        #！！！！！！！！！特征和embedding的融合！！
        # 拼接输入和隐藏状态并通过全连接层
        '''combined_features=torch.cat((inputs_embeds, hidden_states), dim=-1)
        #hidden_states=self.layer(hidden_states)
        # 将x和y的特征进行外积
        gate = torch.sigmoid(self.gate_layer(combined_features))  # [batch_size, seq_len, hidden_size]
        fused_features = gate * inputs_embeds + (1 - gate) * hidden_states  # 加权融合
    
        gate = self.act(self.gate_proj(combined_features))
        value = self.value_proj(combined_features)
        new_hidden_states = gate * value  # GLU门控
        new_hidden_states = self.down_proj(new_hidden_states)
        hidden_states=new_hidden_states+fused_features
        #hidden_states = self.se_layer(hidden_states)
        return hidden_states'''

    def reset_kv(self):
        self.stable_kv = None

#推理中使用到
    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        '''
 draft model生成树状token，这里只支持bs=1的情况
 输入：就是示意图中的整个过程
input_ids维度：[BS,1]
hidden_states维度：[BS,1,hidden_dim]
        
        '''
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        start_time = time.time()

#获取当前的一个 token，维度为（batchsize）,
        sample_token = input_ids[:, -1]
        
 # 存储每个步骤的分数
        scores_list = []
         # 存储每个步骤的父节点索引
        parents_list = []
        # 存储生成的 token
        ss_token = []

#去掉输入序列的第一个 token
# 这里注释掉，不用输入全部的序列了，因为使用了MLP
        #input_ids = input_ids[:, 1:]
        #input_ids = input_ids.to(hidden_states.device)
#len_posi代表当前生成 token 序列的长度：单个token

# 初始化
        len_posi = input_ids.shape[1] # 不依赖历史 token，长度重置为 0
        self.reset()

#保持（bs,1）的维度，取最后一个token 
        input_ids=input_ids[:,-1:]
        hidden_states= hidden_states[:, -1:, :]
        
        #hidden_states= hidden_states[:, -1:, :]
            # 截取前两个时间步的 hidden states
        
        #out_hidden = self(hidden_states=hidden_states, input_ids=input_ids)
        out_hidden = self(
        hidden_states=hidden_states,
        input_ids=input_ids

    )
        '''print(out_hidden.shape)
        print(states["mlstm_state"])
        print(states["conv_state"])'''

#新生成的last_hidden，为了取新生成的最后一个Token.相当于token树生成的根节点。
# 这里相当于length=1，last_hidden 的形状是 (bs=1, hidden_size)
        last_hidden = out_hidden[:, -1]
#计算最后隐藏状态的 logits
        last_headout = head(last_hidden)
#last_p 是经过 logsoftmax 处理后的概率分布（形状为 (1, vocab_size)，即词表中每个 token 的概率）
        last_p = self.logsoftmax(last_headout)

    #topk_index, topk_p：选取 top-k 的 token 和对应的概率值。    
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
    #!!!!!!将 top-1 的对数概率作为初始的 scores 值。!!!!!!!!!!
        scores = topk_p[0]
    #scores[None] 将 scores 转换为 2D tensor，以便与 scores_list 的其他元素维度一致。    
        scores_list.append(scores[None])
     #parents_list 保存生成树的父节点信息，初始化时为 0，表示根节点没有父节点。   
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
    #将 topk_index 保存到 ss_token 中，ss_token 会记录每一层的候选 token 索引。
#这些索引用于在生成树中追踪每个层次的 top-k 候选 token。    
        ss_token.append(topk_index)
    #将 topk_index 赋值给 input_ids，作为下一步输入 token，进行后续层次的生成。    
        input_ids = topk_index
    #last_hidden[None] 增加一个维度，使其可以在 top-k 路径上重复
# repeat(1, top_k, 1) 将其扩展到 [1, top_k, hidden_dim] 的形状。第 1 维重复 top_k 次（即为每个 Top-K 候选 token 分配一份 last_hidden 的副本）。  
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        # 状态也复制 top_k 次
    # 假设 states 中 mlstm_state 与 conv_state 均为张量，形状 [1, hidden_size] 或带候选维度时形状 [1,1, ...]
    # 这里简单地将初始状态复制 top_k 次
        '''candidate_mlstm_state = {}
        for key, value in states["mlstm_state"].items() if isinstance(states["mlstm_state"], dict) else [("mlstm_state", states["mlstm_state"])]:
        # 若是字典形式，则对每个 tensor进行复制
            candidate_mlstm_state[key] = value.unsqueeze(1).repeat(1, top_k, *([1] * (value.ndim - 1)))
        if not candidate_mlstm_state:
        # 若 mlstm_state 本身为 tensor
            candidate_mlstm_state = states["mlstm_state"].unsqueeze(1).repeat(1, top_k, 1)
    # 同理处理 conv_state（这里假设为 tensor）
        candidate_conv_state = states["conv_state"].unsqueeze(1).repeat(1, top_k, 1)'''

        tree_mask = self.tree_mask_init
    #topk_cs_index 保存当前 top-k 候选 token 的索引。
#torch.arange(top_k) 生成从 0 到 top_k - 1 的索引序列，    
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
         # 初始化阶段，增加 current_parent 和 grandparents_list
        
        # 4
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids


    #这里self调用的Model类中的forward， out_hidden的维度(bs,top-k, hidden_size)
            #out_hidden = self(hidden_states=input_hidden, input_ids=input_ids,position_ids=position_ids)   
            out_hidden = self(
            hidden_states=input_hidden,
            input_ids=input_ids,
            position_ids=position_ids

        )

            len_posi += 1
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            #树结构的偏移
            bias = 1 + top_k ** 2 * bias2 + bias1
            #将 topk_cs_index 与bias相加，得到该层生成的父节点索引。
            parents = (topk_cs_index + bias)
            parents_list.append(parents)


#以下是重排阶段！！！！！！！！
#------模型会生成当前层的所有候选 token 的分数，并将这些分数与上一层的累积分数进行相加：
#这里源代码希望直接生成一层，多个候选token
            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
#累加 scores 和当前的 topk_p 得分，得到候选项的累计得分 cu_scores。
# 注意：因为经过的是logsoftmax，因此得分其实是相加，但是在计算逻辑上是相乘,所以是利用先验概率和后验概率进行的
            #!!!当前分数 = 父节点分数 + 当前对数概率
            cu_scores = topk_p + scores[:, None]
 #------------------------------------------           


#--------对累计得分进行重新排序---------
#再次选择 top-k 累计得分候选项。
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            #更新 scores，用于存储当前深度的累计得分。
            scores = topk_cs_p
#----------------------------------------


#-----------重排序后更新下一层的输入------------
#topk_cs_index 中的父级候选项索引，从而获取这些候选项在上一层生成的 token 中的对应位置。
            out_ids = topk_cs_index // top_k            
            input_hidden = out_hidden[:, out_ids]
#基于上一轮生成的 topk_index， 将 topk_index 中的候选项按照 topk_cs_index 的排列方式进行重排，保证每个候选项的 token 序列能反映当前层级的 top-k 候选项。
            input_ids = topk_index.view(-1)[topk_cs_index][None]
#---------------------------------------------------
#             
            # print(input_ids.equal(input_ids0))
#用于最终生成的 token
            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            #更新树掩码，拼接新生成的掩码。通过将上一层候选项的掩码与初始掩码拼接来生成的，确保了每一层的生成结构对树结构的扩展。
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

            #if cu_scores.max() < self.threshold:

            #    break


# 展开为一维张量
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
#获取最终 top scores 和索引。        
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

#提取最终排序的 top-k 候选 token，生成最终的 draft_tokens
        draft_tokens = ss_token_list[top_scores_index]
#将 sample_token 拼接到 draft_tokens 的最前端，确保包含了初始的 token        
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

#合并所有生成步骤中的父节点信息，提取出当前 draft_tokens 的父节点信息
        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        #draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // len_posi].long()
#在 top_scores_index 中找到每个 draft_parents 对应的索引位置，生成新的 mask_index，表示每个 token 的父节点。        
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
# 对没有父节点的 token 进行标记，避免错误连接
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
#mask_index_list 是 mask_index 的列表版本，用于后续树结构计算        
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
# tree_mask初始化为单位矩阵，表示每个 token 对自身可见，并设定第0列为True。
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
#通过 mask_index_list 构建树结构的可见性掩码，使得每个节点能看到其父节点。        
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

# tree_position_ids 表示每个节点在树中的深度（即其父节点路径的长度）。
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

#维度调整
        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

#变量清理
        del parents_list, scores_list, ss_token, ss_token_list, draft_parents


#计算树的最大深度，找出非叶子节点和叶子节点数量，便于后续生成检索索引。
        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

#初始化检索索引
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

#遍历 tree_position_ids，为每个叶节点生成路径回溯（从当前节点到根节点），并将路径存入 retrieve_indices。
        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

#如果启用 logits_processor，根据自定义的 custom_sort 排序规则对 retrieve_indices 进行排序。(自定义排序)
        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)
        
        #print(total_time)
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids




class Vhead(nn.Module):
    def __init__(self, ins=6566, outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins, outs, bias=False)

    def forward(self, x):
        return self.fc(x)


import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = EConfig.from_pretrained('/mnt/EAGLE/eagle/train/MLP/state_16/')
    model = Model(config, load_emb=False)
    print(model)
    #model.topK_genrate

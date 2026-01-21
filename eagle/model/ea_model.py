import copy
import json
import time
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig

import time
import sys
sys.path.append('/root/autodl-fs/EAGLE/eagle/model')
from modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from utils import *
from kv_cache import initialize_past_key_values
from choices import mc_sim_7b_63
from cnets import Model
from configs import EConfig





class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,#draft_model 所生成的令牌总数
            depth,
            top_k,
            threshold, # 阈值
            ea_layer_state_dict #不带模型结构的模型参数
    ):

        super().__init__() # 调用父类构造函数
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        #使用 Hugging Face 的 AutoTokenizer 从预训练模型路径加载词汇表及相关的模型配置。
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path,use_fast=False)
        #eagle模型加载，这个层通过 EA 扩展配置 (EConfig) 被初始化，涉及到特定的深度、top-k 值和阈值
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]#偏置项
        except:
            bias=True # 如果没有，设置默认偏置为 True
        #ea_layer 是draft model部分
        self.ea_layer = Model(config,bias=bias,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)

#低内存模式设置
        low_memory=False
#base_model.model.layers[-1]：基础模型的最后一层；self_attn：自注意力机制；q_proj.weigh：查询（Query）权重投影矩阵
        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        # 判断base_model的自注意力设备与lm_head的设备是否相同
        if device!=base_model.lm_head.weight.device:
             # 设置为不同设备
            self.ea_layer.diff_device = True
            if not low_memory:
            #克隆base_model的lm_hesd的权重，并将其转移到device（base_model的自注意力设备）上
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
    #device记录到self.ea_layer.layer_device中。
    # 目的是在后续计算中可以知道ea_layer的设备是什么，但并不进行权重的克隆，以节省内存。
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
            #加载ea_layer的全部模型参数
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        # 将 EA 层移动到相应设备和数据类型
        self.ea_layer.to(self.base_model.dtype).to(device)
        # 初始化树结构
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer
#支持 LLaMA、Qwen 或 Mixtral
    @classmethod
    def from_pretrained(
            cls, #代表当前类（即 EaModel）
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            total_token=59,
            depth=5,
            top_k=10,
            threshold=1.0,
            **kwargs,# 其他，关键字参数，字典类型
    ):
        #assert Type=="LLaMA" or "Mixtral" 模型架构
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type=='Qwen2ForCausalLM':
            base_model=KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

##加载EA的配置文件
        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
#加载EA层的权重
        try:
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
                #加载权重文件
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )


#如果 total_token 没有给定，模型会自动选择一个最佳的 total_token 值，这个值根据速度和时间来确定。
        if total_token==-1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            # 候选的 total_token 值
            cans=[40,48,50,56,60]
            x=[1,1.05,1.07,1.1,1.13]#调整因子
            times=[]#计算时间

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()  # 同步 CUDA
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids) # 执行前向传播
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token=cans[times.index(min(times))] # 选择最小时间对应的 total_token
            # 设置 EA 层的总 token 数
            model.ea_layer.total_tokens=total_token-1




        return model
    

#模型在推理时如何处理输入。首先，输入通过 base_model 进行处理，并得到隐藏层输出 hidden_states。
#如果需要输出原始结果，output_orig 参数会控制输出原始的 logits 值。
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,# 是否输出原始 logits
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states



        

#@torch.no_grad()是一个装饰器，用于单个函数或方法上，以禁止计算该函数或方法中所有张量操作的梯度。
# 这对于那些不需要梯度的函数或方法非常有用，可以避免不必要的计算。

#eagenerate是基于 EAGLE 机制 的生成方法。利用树形结构来进行推理和采样
    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=1024, # 最大新生成 token 数
            max_length=2048,# 最大生成长度
            log=False,# 是否记录生成过程
            is_llama3=False,

    ):
        if is_llama3:
            #<|eot_id|>：这表示消息在一轮中的结束
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-self.ea_layer.total_tokens-10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        #重置KV缓存
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data



# 得到 draft_tokens（草稿 tokens）、retrieve_indices（检索索引）、tree_mask（树掩码）、tree_position_ids 等数据。这些用于后续的推理和生成。
        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        start_time1=time.time()
#通过 initialize_tree 函数初始化生成树
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        total_time = time.time() - start_time1
       # print("initialize_tree的时间")
       # print(total_time)


        for idx in range(max_length):
            #with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            # 树形解码
            start_time2=time.time()
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            #print(hidden_state_new)
            total_time = time.time() - start_time2
           # print("tree_decoding的时间")
           # print(total_time)
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            #evaluate_posterior根据提供的对数评估候选者的后验概率，并选择最佳候选者。
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            #print(accept_length)
            #with Timer("update_inference_inputs"):
            # update_inference_inputs 函数负责更新推理输入，确保树结构和 KV 缓存的一致性。
            start_time3=time.time()
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            total_time = time.time() - start_time3
           # print("update_inference_inputs的时间")
           # print(total_time)
            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx
    
   


#基于传统方法的生成函数，没有使用 EAGLE 的树形结构，
    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token+=1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx
        
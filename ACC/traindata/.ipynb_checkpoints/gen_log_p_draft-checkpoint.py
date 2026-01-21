from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import json
import torch
from datasets import load_dataset, load_from_disk
import os
import argparse
from tqdm import tqdm
from ast import literal_eval as eval
from util import CKPT, get_model, pretty_format

def read_data(filename):
    data = json.load(open(filename,'r'))
    for item in data: 
        item['prefix'] = eval(item['prefix'])
        item['tokens'] = eval(item['tokens'])
        item['draft'] = eval(item['draft']) 
    return data


@torch.no_grad()
def get_log_prob(data, model):
    for item in data:
        joint = item['prefix'] + item['tokens']
        
        model_device = next(model.parameters()).device
        joint = torch.LongTensor(joint).to(model_device)
        joint = joint.unsqueeze(0)

        index = item['draft'] 
        index = torch.LongTensor(index).to(model_device)
        index = index.unsqueeze(-1)

        
        outputs, last_hidden_states = model(joint, past_key_values=None, output_orig=False)
        batch_size, seq_len, hidden_dim = last_hidden_states.shape
        
        # 创建全零张量用于位置0
        zero_hidden = torch.zeros(
            (batch_size, 1, hidden_dim), 
            device=model_device,
            dtype=last_hidden_states.dtype
        )
        
        # 拼接：位置0使用零向量，后续位置使用前一时间步的隐藏状态
        shifted_hidden = torch.cat([
            zero_hidden,              # 位置0：零向量
            last_hidden_states[:, :-1, :]  # 位置1~n-1：使用0~n-2位置的隐藏状态
        ], dim=1)
        draft_hidden = model.ea_layer(input_ids = joint,hidden_states=shifted_hidden)
        logits=model.base_model.lm_head(draft_hidden)      
        #logits = model(input_ids = joint).logits
        log_probs = logits.log_softmax(dim=-1)  # bs * seq_len * vocab_size
        log_probs_shifted = log_probs[0, len(item['prefix'])-1 : -1] # next_token_log_prob for the continuation

        # take along "item['draft']" 
        log_p = torch.take_along_dim(log_probs_shifted, index, dim=-1) # seq_len * 1
        item[f'log_p_eagle'] = log_p[:, 0].tolist()

    return data

def parse_args():
    parser = argparse.ArgumentParser(description='data generator')

    #parser.add_argument('--model_name', type=str, choices=["7b", "13b", "70b"], default='7b')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str, default=None)

    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = parse_args()
    data = read_data(args.input_file)

    #tokenizer, model = get_model(args.model_name)
    base_model_path="/root/autodl-fs/Llama-2-13b-chat-hf"
    eagle_model_path="/root/autodl-fs/EAGLE/eagle_model/EAGLE-llama2-chat-13B/"
    import sys
    sys.path.append('/root/autodl-fs/EAGLE/eagle/model/') 
    from ea_modelcopy import EaModel  # your local EAGLE model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=eagle_model_path,
        total_token=-1,
        threshold=1.0
    )
    data = get_log_prob(data, model)
    suffix = 'logP'
    if args.output_file is None or len(args.output_file) == 0:
        args.output_file = args.input_file.rstrip('.json') + '_' + args.model_name + suffix + '.json'

    data = pretty_format(data)


    with open(args.output_file, 'w') as f:
        f.write(json.dumps(data, indent=2))

#python gen_log_p_draft.py --input_file /root/autodl-tmp/SpecDec_pp-main/specdec_pp/data/llama_13b_data/tmp3.json --output_file /root/autodl-tmp/SpecDec_pp-main/specdec_pp/data/llama_13b_data/tmp4.json
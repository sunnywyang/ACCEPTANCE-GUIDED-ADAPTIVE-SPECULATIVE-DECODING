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

    return data


@torch.no_grad()
def get_assistant_result(data, assistant_model, do_sample):
    
    for item in data:
        joint = item['prefix'] + item['tokens']
        model_device = next(assistant_model.parameters()).device
        joint = torch.LongTensor(joint).to(model_device)
        joint = joint.unsqueeze(0)
        outputs, orig, last_hidden_states = assistant_model(joint, past_key_values=None, output_orig=True)
        batch_size, seq_len, hidden_dim = last_hidden_states.shape
        

        zero_hidden = torch.zeros(
            (batch_size, 1, hidden_dim), 
            device=model_device,
            dtype=last_hidden_states.dtype
        )
        
        # 拼接：位置0使用零向量，后续位置使用前一时间步的隐藏状态
        shifted_hidden = torch.cat([
            zero_hidden,              # 位置0：零向量
            last_hidden_states[:, :-1, :]
        ], dim=1)
        draft_hidden = assistant_model.ea_layer(input_ids = joint,hidden_states=shifted_hidden)
        sm_logits=assistant_model.base_model.lm_head(draft_hidden)
        if do_sample:
            probs = sm_logits.softmax(dim=-1)  # bs * seq_len * vocab_size
            new_token = torch.multinomial(probs[0], num_samples=1).squeeze(-1)
            item['draft'] = new_token[len(item['prefix'])-1 : -1].tolist()
        else:
            new_token = sm_logits.argmax(dim=-1) # bs * seq_len
            item['draft'] = new_token[0, len(item['prefix'])-1 : -1].tolist()
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='data generator')

    #parser.add_argument('--model_name', type=str, choices=["7b"], default='7b')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--do_sample', action='store_true')

    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = parse_args()
    data = read_data(args.input_file)
    #data = [data[0]] if data else []
    base_model_path="/root/autodl-fs/Llama-2-13b-chat-hf"
    eagle_model_path="/root/autodl-fs/EAGLE/eagle_model/EAGLE-llama2-chat-13B"
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
    #tokenizer, model = get_model(args.model_name)
    data = get_assistant_result(data, model, args.do_sample)

    if args.output_file is None or len(args.output_file) == 0:
        if args.do_sample:
            suffix = 'stochastic'
        else:
            suffix = 'greedy'
        args.output_file = args.input_file.rstrip('.json') + '_' + 'eagle' + suffix + '.json'

    data = pretty_format(data)

    with open(args.output_file, 'w') as f:
        f.write(json.dumps(data, indent=2))


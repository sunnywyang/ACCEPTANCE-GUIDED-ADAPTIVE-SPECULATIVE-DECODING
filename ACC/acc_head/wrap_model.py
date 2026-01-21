import transformers
from transformers import AutoTokenizer, EsmForMaskedLM,  AutoModelForCausalLM, Trainer, TrainingArguments
from tokenizers import Tokenizer
from dataclasses import dataclass, field
from typing import  Optional
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from huggingface_hub import PyTorchModelHubMixin

logger = logging.get_logger(__name__)


class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        #torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))

class AcceptancePredictionHead(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        self.config=config
        hidden_size = config['hidden_size']
        num_layers = config.get('num_layers', 0)
        super().__init__()
        self.model = nn.Sequential( *([ResBlock(hidden_size)] * num_layers), nn.Linear(hidden_size, 2) )

    def forward(self, x):
        return self.model(x)

'''class WrapModel(PreTrainedModel):
    def __init__(self, model, head):
        super().__init__(model.config)
        self.model = model
        self.assist_acc_head = head

    def forward(self, input_ids = None, labels = None, **kwargs):
        return self.model(input_ids = input_ids, labels = labels, **kwargs)'''
class WrapModel(PreTrainedModel):
    def __init__(self, model, head):
        super().__init__(model.config)
        '''self.target_model = target_model  # 原始目标模型
        self.draft_model = draft_model    # Eagle-2草稿模型'''
        self.model = model
        self.assist_acc_head = head       # 接受头

    def forward(self, input_ids=None, labels=None, **kwargs):
        # 使用目标模型获取隐藏状态
        with torch.no_grad():
            outputs,target_hidden_states = self.model(input_ids=input_ids, output_orig=False, **kwargs)
            #target_hidden_states = outputs.hidden_states[-1]
        batch_size, seq_length, hidden_dim = target_hidden_states.shape

# 创建全零的起始向量 (形状 [batch_size, 1, hidden_dim])
        start_vector = torch.zeros(
    batch_size, 1, hidden_dim,
    dtype=target_hidden_states.dtype,
    device=target_hidden_states.device
        )
        target_hidden_states=torch.cat([start_vector, target_hidden_states], dim=1)
        # 使用草稿模型进行预测
        draft_hidden = self.model.ea_layer(
            hidden_states=target_hidden_states[:, :-1, :],
            input_ids=input_ids
        )
        draft_logits=self.model.base_model.lm_head(draft_hidden)
        
        # 返回结果，兼容Trainer
        return draft_logits,draft_hidden
        


if __name__ == "__main__":
    #input_ids = labels = torch.LongTensor([[1,2,3]])
    #model = transformers.AutoModelForCausalLM.from_pretrained("ckpt/hf-llama2-7b-chat")
    #wrapped = WrapModel(model, num_layers=2)
    #AcceptancePredictionHead.from_pretrained('../exp-weight6-layer3')

    import pdb;pdb.set_trace()

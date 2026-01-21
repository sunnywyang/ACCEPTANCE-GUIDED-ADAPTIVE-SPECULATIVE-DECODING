# 在导入部分添加日志配置
import logging
from datetime import datetime

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, TYPE_CHECKING, Any, Callable, Tuple, Union

import torch
import transformers
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import Trainer,AutoTokenizer
import json
import numpy
import scipy.special
from ast import literal_eval as eval

from wrap_model import WrapModel, AcceptancePredictionHead
from transformers import EvalPrediction
import sys
sys.path.append('/root/autodl-fs/EAGLE/eagle/model/') 
from ea_modelcopy import EaModel  # your local EAGLE model
from cnetscopy import Model
from configs import EConfig

import os
os.environ["WANDB_INIT_TIMEOUT"] = "600"  # 增加到300秒


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
#评估指标函数 compute_metrics
# 只会在评估阶段（trainer.evaluate()）被调用，用来计算并汇报定义的 KL 指标

from transformers import TrainerCallback

'''class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:  # 只在主进程记录
            message = ""
            if "loss" in logs:
                message += f"Step {state.global_step} - Loss: {logs['loss']:.4f} "
            if "eval_KL" in logs:  # 注意eval前缀
                message += f"Eval KL: {logs['eval_KL']:.4f}"
            if "KL" in logs:  # 训练中的KL
                message += f"Train KL: {logs['KL']:.4f}"
            
            if message:
                # 使用日志记录器而不是print
                logging.info(message)
                
    # 添加评估结束时的详细记录
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_local_process_zero and metrics:
            logging.info(f"\n===== 评估结果 [Step {state.global_step}] =====")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")
            logging.info("=" * 50)'''
from torch.utils.tensorboard import SummaryWriter

class TensorBoardCallback(TrainerCallback):
    def __init__(self, log_dir="logs"):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            # 记录训练损失
            if "loss" in logs:
                self.writer.add_scalar("train/loss", logs["loss"], state.global_step)
            
            # 记录训练KL
            if "KL" in logs:
                self.writer.add_scalar("train/KL", logs["KL"], state.global_step)
            
            # 记录验证KL
            if "eval_KL" in logs:
                self.writer.add_scalar("eval/KL", logs["eval_KL"], state.global_step)
            
            # 记录学习率
            if "learning_rate" in logs:
                self.writer.add_scalar("train/lr", logs["learning_rate"], state.global_step)
        
        self.step = state.global_step
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_local_process_zero and metrics:
            # 记录所有评估指标
            for key, value in metrics.items():
                self.writer.add_scalar(f"eval/{key}", value, state.global_step)
        
    def __del__(self):
        self.writer.close()

                
def compute_metrics(eval_pred: "EvalPrediction") -> Dict:
    num_class = 2
    # 模型输出的 logits，形状 (batch, seq_len, 2)
    logits= eval_pred[0]
    # # 输入的“软标签”，形状 (batch, seq_len)
    soft_labels = eval_pred[1]

    logits = logits.reshape(-1, num_class)
    soft_labels = soft_labels.reshape(-1)

    not_ignore = (soft_labels - IGNORE_INDEX) > 0.1

    target_prob = soft_labels[not_ignore]
    logits = logits[not_ignore]
    predicted_log_prob = scipy.special.log_softmax(logits, axis=-1)

    # KL divergence:
    CrossEnt = target_prob * ( - predicted_log_prob[:,1]) + (1-target_prob) * ( - predicted_log_prob[:,0])
    Ent = target_prob * numpy.log(target_prob) + (1-target_prob) * numpy.log(1-target_prob)
    Ent[numpy.isnan(Ent)] = 0.  # hack for binary entropy
    KL_binary = CrossEnt - Ent
    KL_binary = numpy.mean(KL_binary)

    return {
        'KL': KL_binary,  # 自动添加eval_前缀
        'custom_KL': KL_binary  # 额外添加无前缀版本
    }


class MyTrainer(Trainer):
#损失计算是由 MyTrainer 里重写的 compute_loss 方法来完成的
    def compute_loss(self, model, inputs, return_outputs=False):
        soft_labels = inputs.pop('soft_labels')
        #创建一个掩码 mask，用于标记哪些位置的标签不是 IGNORE_INDEX，后续计算损失时会忽略 IGNORE_INDEX 对应的位置。
        mask = (soft_labels - IGNORE_INDEX).abs() > 0.1

        #soft_labels_1 即为原始的软标签。
        soft_labels_1 = soft_labels
        #soft_labels_0 是 soft_labels_1 的副本，在掩码位置取反，用于表示负类的软标签。
        soft_labels_0 = soft_labels_1.clone()
        soft_labels_0[mask] = 1 - soft_labels_1[mask]

        label_0 = torch.ones_like(soft_labels, dtype=torch.long).to(soft_labels.device) * IGNORE_INDEX
        label_0[mask] = 0
        label_1 = torch.ones_like(soft_labels, dtype=torch.long).to(soft_labels.device) * IGNORE_INDEX
        label_1[mask] = 1
        
#对draft模型进行前向传播，传入输入数据 inputs，并要求返回隐藏状态。
        '''outputs = model.model(**inputs, output_hidden_states = True)
        hidden_states = outputs
        #！！！！！这里[-1]是最后一层的hidden_states，不是最后一个token的
        #这里的assist_acc_head也是本文的head
        orignal_logits = model.assist_acc_head(hidden_states[-1])'''
        draft_logits,draft_hidden_states = model(**inputs)
        #print(draft_logits)
        #print("OKOKOK")
        #print(draft_hidden_states)
        orignal_logits = model.assist_acc_head(draft_hidden_states)
        orignal_logits = orignal_logits.float()

        num_class = 2

#！！！！！weight_mismatch= 6 for balancing classes，这里的weight_mismatch就是wrej
        weight = torch.tensor([self.args.weight_mismatch, 1]).to(orignal_logits.device)
        loss_fct = CrossEntropyLoss(weight=weight, reduction='none')

        logits = orignal_logits.view(-1, num_class)
        label_0 = label_0.view(-1)
        label_1 = label_1.view(-1)
        soft_labels_0 = soft_labels_0.view(-1)
        soft_labels_1 = soft_labels_1.view(-1)
        mask = mask.view(-1)

        loss_0 = loss_fct(logits, label_0) # (bs * seq_len), num_class
        loss_1 = loss_fct(logits, label_1) # (bs * seq_len), num_class

        # reduce with soft labels, coresponding to BCELoss
        loss = (loss_0 * soft_labels_0 + loss_1 * soft_labels_1).sum() / (self.args.weight_mismatch * soft_labels_0[mask].sum() +  soft_labels_1[mask].sum() )
        
        # 如果是训练模式，额外计算并记录 KL 散度指标
        if model.training:
            # KL divergence:
            target_prob = soft_labels_1[mask]
            predicted_logits = logits[mask, :]
            predicted_log_prob = torch.log_softmax(predicted_logits, dim=-1)

            #KL_binary = target_prob * (target_prob.log() - predicted_log_prob[:,1]) + (1-target_prob) * ( (1-target_prob).log() - predicted_log_prob[:,0])

            CrossEnt = target_prob * ( - predicted_log_prob[:,1]) + (1-target_prob) * ( - predicted_log_prob[:,0])
            Ent = target_prob * target_prob.log() + (1-target_prob) * (1-target_prob).log()
            Ent[Ent.isnan()] = 0.  # hack for binary entropy
            KL_binary = CrossEnt - Ent
            KL_binary = KL_binary.mean().item()

            self.log({'train_KL': KL_binary})

        if return_outputs:
            outputs = (loss, orignal_logits)
            return (loss, outputs)
        else:
            return loss

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bf16: bool = True
   # model_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None)
    logging_dir: Optional[str] = field(default="/root/tf-logs/")  # 日志目录
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])  # 报告到TensorBoard
    logging_strategy: str = field(default="steps")      # 按步记录
    logging_steps: int = field(default=50)             # 每50步记录一次
    evaluation_strategy: str = field(default="steps")   # 按步评估
    eval_steps: int = field(default=100)               # 每200步评估一次
    #report_to: List[str] = field(default_factory=list) # 禁用第三方报告
    eval_data_path: str = field(default=None)
    remove_unused_columns: bool = False
    evaluate_only: bool = False
    label_names: Optional[List[str]] = field(
        default_factory=lambda: ['soft_labels'], metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )

    weight_mismatch: Optional[float] = field(default = 1.) # 6 for balancing classes
    resnet_num_layers: Optional[int] = field(default = 1)
    mixing_ratio: Optional[float] = field(default = 0.15)
    

#当添加新 special token（如 [PAD]）时，
# 自动扩容模型的词表和 embedding，并用现有 embedding 的均值初始化新增部分。
'''def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg'''



class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, r: float = 0.15):
        super(SupervisedDataset, self).__init__()
        logging.warning(f"Loading data... from {data_path}")
        data = json.load(open(data_path,'r'))
        self.input_ids = []
        self.soft_labels = []
        for item in data:
            item['prefix'] = eval(item['prefix'])
            item['tokens'] = eval(item['tokens'])
            item['draft'] = eval(item['draft'])

            # item['tokens'] are generated autoregressively from target model
            # item['draft'] are stochatic next-token predicted by the draft model

            item['p_acc'] = eval(item['p_acc'])

            prefix = torch.LongTensor(item['prefix'])
            Xs = torch.LongTensor(item['tokens'])
            # Ys = torch.LongTensor(item['draft'])

            # take r from Xs and (1-r) from Ys.
            mask = (torch.rand(*Xs.shape) < r)
            Zs = torch.LongTensor(item['draft'])
            Zs[mask] = Xs[mask]

            self.input_ids.append(torch.cat([prefix, Zs]))

            label_prefix = torch.tensor([IGNORE_INDEX] * len(item['prefix']))
            p_acc = torch.tensor(item['p_acc'])

            # don't calculate loss on Xs.
            p_acc[mask] = IGNORE_INDEX

            self.soft_labels.append(torch.cat([label_prefix, p_acc]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], soft_labels=self.soft_labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, soft_labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "soft_labels"))
        # 确保使用整数作为填充值
        pad_token_id =  0
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        soft_labels = torch.nn.utils.rnn.pad_sequence(soft_labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            soft_labels=soft_labels,
            attention_mask=input_ids.ne(pad_token_id).int(),
        )




if __name__ == "__main__":
    parser = transformers.HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()[0]
    # 禁用 WandB 报告
    '''training_args.report_to = []
    
    # ====== 新增：配置日志系统 ======
    log_dir = training_args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler()         # 输出到控制台
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("===== 开始训练 =====")
    logger.info(f"日志文件: {log_file}")'''
    # ====== 日志配置结束 ======
    
    # 确保日志目录存在
    os.makedirs(training_args.logging_dir, exist_ok=True)
    
    # 创建TensorBoard回调
    tb_callback = TensorBoardCallback(log_dir=training_args.logging_dir)

    '''tokenizer = transformers.AutoTokenizer.from_pretrained(training_args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(training_args.model_name_or_path)'''
    base_model_path="/root/autodl-fs/Llama27bchathf/"
    eagle_model_path="/root/autodl-fs/EAGLE/eagle_model/EAGLE-llama2-chat-7B/"
    import sys
    sys.path.append('/root/autodl-fs/EAGLE/eagle/model/') 
    from ea_modelcopy import EaModel  # your local EAGLE model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # 添加以下代码确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=eagle_model_path,
        total_token=-1,
        threshold=1.0
    )
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    '''smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model.base_model,
    )'''

    train_dataset = SupervisedDataset(training_args.data_path, r=training_args.mixing_ratio)
    if training_args.eval_data_path is not None:
        eval_dataset = SupervisedDataset(training_args.eval_data_path, r=training_args.mixing_ratio)
        print("num eval example:", len(eval_dataset))
    else:
        eval_dataset = None
    vocab_size = tokenizer.vocab_size
    # 只看前 100 个样本作为示例，或你也可以看全量
    for split_name, ds in [("train", train_dataset), ("eval", eval_dataset)]:
        if ds is None:
            continue
        for i in range(min(len(ds), 10000)):
            ids = ds[i]["input_ids"]
            if int(ids.max().item()) >= vocab_size:
                raise RuntimeError(
                    f"在 {split_name} 集合中，第 {i} 条样本出现越界 token id = {int(ids.max().item())} (≥ vocab_size={vocab_size})"
                )
    print("Sanity check: 所有输入 token id 都在 [0, vocab_size) 范围内")
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    acc_head_config = {'hidden_size': model.config.hidden_size, 'num_layers': training_args.resnet_num_layers}
    assist_acc_head = AcceptancePredictionHead(acc_head_config)
    wrapped = WrapModel(model, assist_acc_head)
    wrapped.model.requires_grad_(False)
    print('num training example:', len(train_dataset))
    '''trainer = MyTrainer(model=wrapped, tokenizer=tokenizer, args=training_args, train_dataset = train_dataset, eval_dataset = eval_dataset, data_collator=data_collator, compute_metrics = compute_metrics)'''
        # 在创建 Trainer 时添加回调
    trainer = MyTrainer(
        model=wrapped,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[tb_callback]  # 添加回调
    )
    if training_args.evaluate_only:
        print("eval only. Loading from checkpoint:", training_args.output_dir)
        wrapped.assist_acc_head = AcceptancePredictionHead.from_pretrained(training_args.output_dir)
        trainer.evaluate()
    else:
        trainer.train()
        trainer.save_state()
        wrapped.assist_acc_head.save_pretrained(training_args.output_dir, config=acc_head_config)

        # python3 train-llama7b.py     --data_path /root/autodl-fs/SpecDec_pp-main/llama_7b_5W2/train.json    --eval_data_path /root/autodl-fs/SpecDec_pp-main/llama_7b_5W2/dev.json     --output_dir llama7b-layer1-mixing0.5    --bf16 True     --per_device_train_batch_size 4    --num_train_epochs 3     --gradient_accumulation_steps 8     --logging_steps 5     --evaluation_strategy epoch     --per_device_eval_batch_size 4     --weight_mismatch 6.0     --save_strategy no     --warmup_ratio 0.03    --lr_scheduler_type cosine     --resnet_num_layers 1     --mixing_ratio 0.5
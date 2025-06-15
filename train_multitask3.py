import json
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    BertPreTrainedModel,
    BertModel,
    BertTokenizerFast,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics import f1_score, precision_score, recall_score

# --- 1. 全局配置和标签定义 ---
# 使用一个常见的中文预训练模型
MODEL_NAME = "autodl-tmp/chinese-bert-wwm-ext" 
OUTPUT_DIR = "autodl-tmp/hate-speech-multitask-model"

# 定义分类标签到ID的映射
# 注意：我们将所有原始标签统一为小写，以避免大小写不一致问题
GROUP_LABELS_LIST = sorted(["Racism", "Region", "Sexism", "LGBTQ", "others", "non-hate"])
HATEFUL_LABELS_LIST = sorted(["hate", "non-hate"])

GROUP_TO_ID = {label: i for i, label in enumerate(GROUP_LABELS_LIST)}
ID_TO_GROUP = {i: label for i, label in enumerate(GROUP_LABELS_LIST)}

HATEFUL_TO_ID = {label: i for i, label in enumerate(HATEFUL_LABELS_LIST)}
ID_TO_HATEFUL = {i: label for i, label in enumerate(HATEFUL_LABELS_LIST)}

# --- 2. 数据解析与扁平化 ---
def parse_and_flatten_data(raw_data):
    """
    解析原始JSON数据，将其扁平化为每个四元组一条记录。
    处理包含多个类别的情况，如 "Sexism, Racism"，将其拆分为两条独立记录。
    """
    flattened_data = []
    print("Parsing and flattening raw data...")
    for entry in tqdm(raw_data):
        content = entry["content"]
        output_str = entry["output"]

        # 去掉结尾的 [END] 并按 [SEP] 分割成多个四元组字符串
        quadruplets_str = output_str.strip().removesuffix("[END]").strip()
        quad_parts = [q.strip() for q in quadruplets_str.split("[SEP]")]

        for part in quad_parts:
            elements = [e.strip() for e in part.split("|")]
            if len(elements) != 4:
                print(f"Skipping malformed entry: {part} in ID {entry['id']}")
                continue

            target, argument, group_str, hateful = elements
            
            # 统一标签为小写
            # hateful = hateful.lower()
            
            # 处理可能存在的多个group标签
            groups = [g.strip() for g in group_str.split(',')]
            
            for group in groups:
                if group not in GROUP_TO_ID:
                    print(f"Warning: Unknown group label '{group}' found in ID {entry['id']}. Mapping to 'others'.")
                    group = 'others'

                flattened_data.append({
                    "id": entry["id"],
                    "content": content,
                    "target": target,
                    "argument": argument,
                    "group": group,
                    "hateful": hateful
                })
    return flattened_data

# --- 3. 自定义多任务模型 ---
class BertForMultiTaskHateDetection(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_group_labels = len(GROUP_TO_ID)
        self.num_hateful_labels = len(HATEFUL_TO_ID)

        # 核心BERT模型
        self.bert = BertModel(config, add_pooling_layer=False)

        # 任务头
        # a) 抽取头 (两个独立的QA头)
        self.target_qa_outputs = nn.Linear(config.hidden_size, 2)
        self.argument_qa_outputs = nn.Linear(config.hidden_size, 2)
        
        # b) 分类头
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.group_classifier = nn.Linear(config.hidden_size, self.num_group_labels)
        self.hateful_classifier = nn.Linear(config.hidden_size, self.num_hateful_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        target_start_positions=None,
        target_end_positions=None,
        argument_start_positions=None,
        argument_end_positions=None,
        group_labels=None,
        hateful_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # [batch_size, seq_len, hidden_size]
        sequence_output = outputs[0]
        # [batch_size, hidden_size] for [CLS] token
        cls_output = sequence_output[:, 0]
        
        # --- 1. 抽取任务 ---
        target_logits = self.target_qa_outputs(sequence_output)
        target_start_logits, target_end_logits = target_logits.split(1, dim=-1)
        target_start_logits = target_start_logits.squeeze(-1).contiguous()
        target_end_logits = target_end_logits.squeeze(-1).contiguous()

        argument_logits = self.argument_qa_outputs(sequence_output)
        argument_start_logits, argument_end_logits = argument_logits.split(1, dim=-1)
        argument_start_logits = argument_start_logits.squeeze(-1).contiguous()
        argument_end_logits = argument_end_logits.squeeze(-1).contiguous()

        # --- 2. 分类任务 ---
        pooled_output = self.dropout(cls_output)
        group_logits = self.group_classifier(pooled_output)
        hateful_logits = self.hateful_classifier(pooled_output)

        # --- 3. 计算总损失 ---
        total_loss = None
        if all(label is not None for label in [target_start_positions, target_end_positions, 
                                               argument_start_positions, argument_end_positions, 
                                               group_labels, hateful_labels]):
            loss_fct = CrossEntropyLoss()
            
            # 抽取损失
            target_start_loss = loss_fct(target_start_logits, target_start_positions)
            target_end_loss = loss_fct(target_end_logits, target_end_positions)
            target_qa_loss = (target_start_loss + target_end_loss) / 2

            argument_start_loss = loss_fct(argument_start_logits, argument_start_positions)
            argument_end_loss = loss_fct(argument_end_logits, argument_end_positions)
            argument_qa_loss = (argument_start_loss + argument_end_loss) / 2
            
            # 分类损失
            group_loss = loss_fct(group_logits, group_labels)
            hateful_loss = loss_fct(hateful_logits, hateful_labels)

            # 组合损失 (可以给不同任务分配权重，这里简单相加)
            total_loss = target_qa_loss + argument_qa_loss + group_loss + hateful_loss

        return {
            "loss": total_loss,
            "target_start_logits": target_start_logits,
            "target_end_logits": target_end_logits,
            "argument_start_logits": argument_start_logits,
            "argument_end_logits": argument_end_logits,
            "group_logits": group_logits,
            "hateful_logits": hateful_logits,
        }

# 将 Trainer 替换为自定义子类，允许传递多个 label 字段
class MultiLabelTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 标准预测流程
        has_labels = all(k in inputs for k in [
            "target_start_positions", "target_end_positions",
            "argument_start_positions", "argument_end_positions",
            "group_labels", "hateful_labels"
        ])
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)

        if prediction_loss_only:
            return (outputs["loss"], None, None)

        logits = (
            outputs["target_start_logits"],
            outputs["target_end_logits"],
            outputs["argument_start_logits"],
            outputs["argument_end_logits"],
            outputs["group_logits"],
            outputs["hateful_logits"],
        )

        if has_labels:
            labels = (
                inputs["target_start_positions"].detach().cpu(),
                inputs["target_end_positions"].detach().cpu(),
                inputs["argument_start_positions"].detach().cpu(),
                inputs["argument_end_positions"].detach().cpu(),
                inputs["group_labels"].detach().cpu(),
                inputs["hateful_labels"].detach().cpu(),
            )

        else:
            labels = None

        return (outputs["loss"].detach().cpu(), logits, labels)

# --- 4. 数据预处理函数 ---
# 实例化分词器，以便在预处理函数中使用
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
MAX_LENGTH = 512

def find_token_indices(text, entity, offset_mapping):
    """在token列表中找到实体对应的start和end token索引"""
    if entity == "NULL" or not entity:
        return 0, 0
    
    start_char = text.find(entity)
    if start_char == -1:
        return 0, 0 # 如果找不到，返回(0, 0)指向[CLS]
    
    end_char = start_char + len(entity)
    
    token_start_index = 0
    token_end_index = 0

    for idx, (offset_start, offset_end) in enumerate(offset_mapping):
        if offset_start <= start_char < offset_end:
            token_start_index = idx
        if offset_start < end_char <= offset_end:
            token_end_index = idx
            break # 找到结尾就可以停止了
            
    # 如果实体跨越多个token但结束token未找到，或者开始token未找到，则返回(0,0)
    if token_start_index == 0 or token_end_index == 0:
        return 0, 0
        
    return token_start_index, token_end_index

# 添加评估指标函数compute_similarity()和compute_custom_metrics()
def compute_custom_metrics(eval_preds):
    def compute_similarity(pred, gold):
        matcher = SequenceMatcher(None, pred, gold)
        match_len = sum(triple.size for triple in matcher.get_matching_blocks())
        return (2 * match_len) / (len(pred) + len(gold)) if (len(pred) + len(gold)) > 0 else 0

    predictions, label_ids = eval_preds
    (
        target_start_logits,
        target_end_logits,
        argument_start_logits,
        argument_end_logits,
        group_logits,
        hateful_logits,
    ) = predictions

    (
        target_start_positions,
        target_end_positions,
        argument_start_positions,
        argument_end_positions,
        group_labels,
        hateful_labels,
    ) = label_ids

    pred_group = np.argmax(group_logits, axis=1)
    pred_hate = np.argmax(hateful_logits, axis=1)

    group_f1 = f1_score(group_labels, pred_group, average="macro")
    hate_f1 = f1_score(hateful_labels, pred_hate, average="macro")
    avg_f1 = (group_f1 + hate_f1) / 2

    # 软匹配抽取评估（按位置对比）
    seq_len = target_start_logits.shape[1]
    pred_target_start = np.argmax(target_start_logits, axis=1)
    pred_target_end = np.argmax(target_end_logits, axis=1)
    pred_argument_start = np.argmax(argument_start_logits, axis=1)
    pred_argument_end = np.argmax(argument_end_logits, axis=1)

    extract_correct = 0
    total = len(pred_group)

    for i in range(total):
        ts_pred, te_pred = pred_target_start[i], pred_target_end[i]
        ts_true, te_true = target_start_positions[i], target_end_positions[i]
        as_pred, ae_pred = pred_argument_start[i], pred_argument_end[i]
        as_true, ae_true = argument_start_positions[i], argument_end_positions[i]

        target_sim = compute_similarity(str(ts_pred) + "-" + str(te_pred),
                                        str(ts_true) + "-" + str(te_true))
        argument_sim = compute_similarity(str(as_pred) + "-" + str(ae_pred),
                                          str(as_true) + "-" + str(ae_true))
        if target_sim > 0.5 and argument_sim > 0.5:
            extract_correct += 1

    extract_f1 = extract_correct / total

    return {
        "group_f1": group_f1,
        "hate_f1": hate_f1,
        "avg_f1": avg_f1,
        "extract_f1": extract_f1
    }

def preprocess_function(example):
    """
    对单条扁平化数据进行处理，生成模型输入。
    """
    content = example['content']
    target_text = example['target']
    argument_text = example['argument']
    
    # Tokenize
    inputs = tokenizer(
        content,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")
    
    # 找到 Target 和 Argument 的 token 索引
    target_start, target_end = find_token_indices(content, target_text, offset_mapping)
    argument_start, argument_end = find_token_indices(content, argument_text, offset_mapping)

    # 添加位置标签
    inputs["target_start_positions"] = target_start
    inputs["target_end_positions"] = target_end
    inputs["argument_start_positions"] = argument_start
    inputs["argument_end_positions"] = argument_end

    # 添加分类标签
    inputs["group_labels"] = GROUP_TO_ID[example['group']]
    inputs["hateful_labels"] = HATEFUL_TO_ID[example['hateful']]
    
    return inputs

# --- 5. 主训练流程 ---
if __name__ == "__main__":
    # 加载原始数据
    try:
        with open("autodl-tmp/train.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("错误: train.json 文件未找到。请确保该文件在当前目录下。")
        exit()

    # 解析并扁平化数据
    flat_data = parse_and_flatten_data(raw_data)
    print(f"\n数据解析完成，共得到 {len(flat_data)} 条独立的四元组训练样本。")
    print("示例数据:", flat_data[0])

    # 创建 Hugging Face Dataset
    full_dataset = Dataset.from_list(flat_data)
    
    # 应用预处理
    print("\n开始预处理数据...")
    processed_dataset = full_dataset.map(preprocess_function, batched=False, num_proc=4,
                                         remove_columns=full_dataset.column_names)
    
    # 划分训练集和验证集
    train_test_split = processed_dataset.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'eval': train_test_split['test']
    })
    print("\n数据集准备就绪:")
    print(dataset_dict)

    # 初始化模型
    print("\n初始化模型...")
    model = BertForMultiTaskHateDetection.from_pretrained(MODEL_NAME)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,  # 建议先从3-5个epoch开始
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_strategy="steps",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="avg_f1",
        greater_is_better=False,
        report_to="none", # 如果需要，可以设置为 "tensorboard" 或 "wandb"
    )

    # 初始化 Trainer
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["eval"],
        tokenizer=tokenizer,
        compute_metrics=compute_custom_metrics,
    )

    # 开始训练
    print("\n--- 开始训练 ---")
    trainer.train()

    # 保存最佳模型
    best_model_path = f"{OUTPUT_DIR}/best"
    print(f"\n训练完成！正在保存最佳模型到 {best_model_path}")
    trainer.save_model(best_model_path)
    print("模型保存完毕。")
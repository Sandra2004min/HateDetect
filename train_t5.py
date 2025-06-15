import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import numpy as np
import evaluate # 使用 evaluate 库来加载 rouge

# --- 1. 全局配置 ---
# 推荐使用孟子T5，这是一个强大的中文T5模型
MODEL_NAME = "autodl-tmp/mengzi-t5-base" 
OUTPUT_DIR = "autodl-tmp/hate-speech-t5-model"

# T5模型的输入需要一个前缀来告诉它任务是什么，这非常重要！
SOURCE_PREFIX = "仇恨言论四元组抽取: "
MAX_SOURCE_LENGTH = 512  # 输入最大长度
MAX_TARGET_LENGTH = 128  # 输出最大长度

# --- 2. 数据预处理函数 ---
def preprocess_function(examples, tokenizer):
    """
    为T5模型准备输入和输出。
    """
    inputs = [SOURCE_PREFIX + doc for doc in examples["content"]]
    targets = [doc for doc in examples["output"]]

    # 对输入进行编码
    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, truncation=True)
    
    # 对目标（标签）进行编码
    # 使用 text_target 参数，tokenizer会自动处理
    labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- 3. 评估指标计算函数 ---
# 我们使用 ROUGE 分数来评估生成文本的质量
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    # 如果模型返回的不是元组，直接使用
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # 将 -100 替换为 pad_token_id，以便解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # 解码预测和标签
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 去掉句子前后的空格
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # 计算 ROUGE 分数
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # 提取主要的ROUGE分数
    result = {key: value * 100 for key, value in result.items()}
    
    # 添加一个预测长度的度量
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# --- 4. 主训练流程 ---
if __name__ == "__main__":
    # 加载原始数据
    try:
        with open("autodl-tmp/train.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("错误: train.json 文件未找到。请确保该文件位于 'autodl-tmp/' 目录下。")
        exit()

    print(f"原始数据加载完成，共 {len(raw_data)} 条。")

    # 创建 Hugging Face Dataset
    full_dataset = Dataset.from_list(raw_data)
    
    # 加载分词器
    print(f"正在从 '{MODEL_NAME}' 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 应用预处理
    print("开始预处理数据...")
    tokenized_dataset = full_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=full_dataset.column_names
    )
    
    # 划分训练集和验证集
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'eval': train_test_split['test']
    })
    print("\n数据集准备就绪:")
    print(dataset_dict)

    # 初始化模型
    print(f"\n正在从 '{MODEL_NAME}' 初始化模型...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # 数据整理器，用于动态填充批次中的序列
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id # 使用 tokenizer 的 pad token id
    )

    # 设置训练参数
    # 使用 Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,  # 建议5-10个epoch
        per_device_train_batch_size=4, # T5模型较大，batch size要小一点
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2, # 梯度累积，等效于 batch_size=8
        learning_rate=3e-5,
        warmup_steps=300,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_strategy="steps",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL", # 使用ROUGE-L作为最佳模型的指标
        greater_is_better=True,
        predict_with_generate=True, # 必须开启，以便在评估时使用 model.generate()
        report_to="none",
    )

    # 初始化 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    print("\n--- 开始训练 ---")
    trainer.train()

    # 保存最佳模型
    best_model_path = f"{OUTPUT_DIR}/best"
    print(f"\n训练完成！正在保存最佳模型到 {best_model_path}")
    trainer.save_model(best_model_path)
    print("模型保存完毕。")
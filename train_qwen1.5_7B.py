import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from tqdm.auto import tqdm

# --- 1. 全局配置 ---
MODEL_NAME = "autodl-tmp/Qwen1.5-7B-Chat" 
OUTPUT_DIR = "autodl-tmp/hate-speech-qwen1.5-7B-lora"
JSON_PATH = "autodl-tmp/train.json"

# --- 2. 数据处理与格式化 (核心改动在这里) ---
def parse_and_format_multilabel(raw_data):
    """
    解析原始JSON，处理多类别标签，并保持原始大小写。
    """
    processed_data = []
    print("Parsing and formatting data, preserving original case for labels...")
    for entry in tqdm(raw_data):
        content = entry["content"]
        output_str = entry["output"]

        # 使用兼容Python 3.8的写法
        temp_str = output_str.strip()
        if temp_str.endswith("[END]"):
            quadruplets_str = temp_str[:-len("[END]")].strip()
        else:
            quadruplets_str = temp_str
        
        quad_parts = [q.strip() for q in quadruplets_str.split("[SEP]")]

        modified_quads = []
        for part in quad_parts:
            elements = [e.strip() for e in part.split("|")]
            if len(elements) != 4:
                continue

            target, argument, group_str, hateful = elements
            
            # 【关键改动】: 不再转换为小写，只做清洗和排序
            groups_raw = [g.strip() for g in group_str.split(',')]
            
            # 清洗可能的拼写错误，但保持大小写
            cleaned_groups = set()
            for g in groups_raw:
                # 示例：如果 'sexis' 存在，则替换为 'Sexism'，保持大小写风格
                if g.lower() == 'sexis':
                    # 根据原始数据中 'Sexism' 的常见大小写来决定
                    # 假设 'Sexism' 是标准格式
                    cleaned_groups.add('Sexism') 
                else:
                    cleaned_groups.add(g)
            
            # 排序以保证输出顺序的一致性
            sorted_groups = sorted(list(cleaned_groups))
            
            # 重新组合成标准字符串
            final_group_str = ", ".join(sorted_groups)
            
            # hateful 标签也不再转为小写
            modified_quads.append(f"{target} | {argument} | {final_group_str} | {hateful}")
        
        final_output = " [SEP] ".join(modified_quads) + " [END]"
        
        processed_data.append({
            "content": content,
            "output": final_output
        })
        
    return processed_data

def format_for_sft(example, tokenizer):
    """
    为SFTTrainer准备最终的文本格式 (使用ChatML模板)。
    """
    system_prompt = (
        "你是一个专业的仇恨言论分析专家。你的任务是从给定的社交媒体文本中，"
        "识别并抽取出所有仇恨言论四元组，并严格按照 'Target | Argument | Group | Hateful' 的格式输出。"
        "其中 'Group' 字段可能包含用逗号分隔的多个类别。"
    )
    user_input = f"请分析以下文本：\n{example['content']}"
    assistant_response = example['output']
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_response}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    return {"text": formatted_text}

# --- 3. 主训练流程 (与之前一致) ---
if __name__ == "__main__":
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: {JSON_PATH} 文件未找到。")
        exit()
    print(f"原始数据加载完成，共 {len(raw_data)} 条。")

    formatted_data = parse_and_format_multilabel(raw_data)
    full_dataset = Dataset.from_list(formatted_data)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    sft_ready_dataset = full_dataset.map(
        lambda example: format_for_sft(example, tokenizer), 
        remove_columns=list(full_dataset.features)
    )

    dataset = sft_ready_dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"正在从 '{MODEL_NAME}' 加载模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # 可以注释掉下一行
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        tf32=True,
        max_steps=-1,
        warmup_ratio=0.03,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        tokenizer=tokenizer,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=1024,
    )

    print(f"\n--- 开始QLoRA微调 ({MODEL_NAME}, 保持原始大小写) ---")
    trainer.train()

    final_model_path = f"{OUTPUT_DIR}/final"
    print(f"\n训练完成！正在保存LoRA适配器到 {final_model_path}")
    trainer.save_model(final_model_path)
    print("适配器保存完毕。")
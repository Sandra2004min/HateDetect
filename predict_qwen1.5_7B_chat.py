import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm.auto import tqdm

def predict_multilabel():
    # --- 配置 ---
    BASE_MODEL_NAME = "autodl-tmp/Qwen1.5-7B-Chat"
    LORA_ADAPTER_PATH = "autodl-tmp/hate-speech-qwen1.5-7B-lora/final"
    TEST_JSON_PATH = "autodl-tmp/test1.json"
    OUTPUT_TXT_PATH = "prediction_qwen1.5_7B.txt"

    # --- 加载模型和分词器 ---
    print(f"正在从 '{BASE_MODEL_NAME}' 加载基础模型和分词器...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    if "[END]" not in tokenizer.vocab:
        tokenizer.add_special_tokens({"eos_token": "[END]"})
    else:
        tokenizer.eos_token = "[END]"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"正在从 '{LORA_ADAPTER_PATH}' 加载并合并LoRA适配器...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    model = model.merge_and_unload()
    
    device = model.device
    model.eval()
    print("模型加载并合并完成，已移至", device)

    # --- 加载测试数据 ---
    try:
        test_df = pd.read_json(TEST_JSON_PATH, lines=True, dtype={'id': str, 'content': str})
    except ValueError:
        test_df = pd.read_json(TEST_JSON_PATH, dtype={'id': str, 'content': str})

    # --- 推理和生成结果 ---
    print(f"开始处理测试数据，结果将保存到 {OUTPUT_TXT_PATH}...")
    with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f_out:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            content = row['content'].strip()
            if not content:
                f_out.write("NULL | NULL | non-hate | non-hate [END]\n")
                continue

            system_prompt = (
                "你是一个专业的仇恨言论分析专家。你的任务是从给定的社交媒体文本中，"
                "识别并抽取出所有仇恨言论四元组，并严格按照 'Target | Argument | Group | Hateful' 的格式输出。"
                "其中 'Group' 字段可能包含用逗号分隔的多个类别。"
            )
            user_input = f"请分析以下文本：\n{content}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            text_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=256,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            if '[END]' in response:
                response = response.split('[END]')[0].strip() + ' [END]'
            else:
                response = response.strip() + ' [END]'

            if response.count('|') < 3:
                f_out.write("NULL | NULL | non-hate | non-hate [END]\n")
            else:
                f_out.write(response + "\n")

    print(f"\n推理完成！预测结果已保存至 {OUTPUT_TXT_PATH}。")

if __name__ == "__main__":
    predict_multilabel()
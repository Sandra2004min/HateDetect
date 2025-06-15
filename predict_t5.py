import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

def predict():
    # --- 配置 ---
    # 指向你训练好的最佳模型文件夹
    MODEL_PATH = "autodl-tmp/hate-speech-t5-model/best"
    TEST_JSON_PATH = "autodl-tmp/test1.json"
    OUTPUT_TXT_PATH = "prediction.txt"

    SOURCE_PREFIX = "仇恨言论四元组抽取: "
    MAX_SOURCE_LENGTH = 512
    MAX_TARGET_LENGTH = 128  # 生成的最大长度

    # --- 加载模型和分词器 ---
    print(f"正在从 {MODEL_PATH} 加载模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"错误：模型路径 {MODEL_PATH} 不存在或不完整。请确保已成功运行训练脚本。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("模型加载完成，已移至", device)

    # --- 加载测试数据 ---
    print(f"正在加载测试数据 {TEST_JSON_PATH}...")
    try:
        test_df = pd.read_json(TEST_JSON_PATH, lines=True, dtype={'id': str, 'content': str})
    except (ValueError, FileNotFoundError):
        try:
            test_df = pd.read_json(TEST_JSON_PATH, dtype={'id': str, 'content': str})
        except Exception as e:
            print(f"无法加载测试数据 {TEST_JSON_PATH}: {e}")
            return

    # --- 推理和生成结果 ---
    print(f"开始处理测试数据，结果将保存到 {OUTPUT_TXT_PATH}...")
    with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f_out:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            content = row['content'].strip()
            if not content:
                # 对于空内容，直接写入默认输出
                f_out.write("NULL | NULL | non-hate | non-hate [END]\n")
                continue

            # 准备模型输入
            input_text = SOURCE_PREFIX + content
            inputs = tokenizer(
                input_text,
                max_length=MAX_SOURCE_LENGTH,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            # 使用 beam search 生成结果，效果通常更好
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=5,  # beam search 的宽度
                    early_stopping=True
                )
            
            # 解码并清理
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 有时模型可能生成空字符串，提供一个默认值
            if not generated_text.strip():
                 f_out.write("NULL | NULL | non-hate | non-hate [END]\n")
            else:
                 f_out.write(generated_text + "\n")

    print(f"\n推理完成！预测结果已保存至 {OUTPUT_TXT_PATH}。")

if __name__ == "__main__":
    predict()
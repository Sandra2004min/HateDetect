# vllm_infer_qwen.py

from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm

def main():
    MODEL_PATH = "./models_ym/qwen3-8B-lora-merged"
    TEST_JSON_PATH = "./data/test1.json"
    OUTPUT_TXT_PATH = "prediction_vllm_qwen3.txt"
    BATCH_SIZE = 16

    # 初始化 vLLM 模型
    print("Loading model with vLLM...")
    llm = LLM(model=MODEL_PATH, trust_remote_code=True, tensor_parallel_size=2)  # 两张 A6000

    # 构造输入
    try:
        df = pd.read_json(TEST_JSON_PATH, lines=True, dtype={'id': str, 'content': str})
    except ValueError:
        df = pd.read_json(TEST_JSON_PATH, dtype={'id': str, 'content': str})

    prompts = []
    for content in df['content']:
        if not content or not content.strip():
            prompts.append(None)
            continue

        system_prompt = (
            "你是一个专业的仇恨言论分析专家。你的任务是从给定的社交媒体文本中，"
            "识别并抽取出所有仇恨言论四元组，并严格按照 'Target | Argument | Group | Hateful' 的格式输出。"
            "其中 'Group' 字段可能包含用逗号分隔的多个类别。"
        )
        user_input = f"请分析以下文本：\n{content.strip()}"
        text_input = (
            f"### Instruction:\n{system_prompt}\n\n"
            f"### Input:\n{user_input}\n\n"
            f"### Response:\n"
        )
        prompts.append(text_input)

    # 设置采样参数（非采样生成更稳定）
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.0,
        stop=["[END]"],
    )

    # 批量推理
    print("Running inference...")
    with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            valid_prompts = [p for p in batch_prompts if p is not None]

            if not valid_prompts:
                for _ in batch_prompts:
                    f_out.write("NULL | NULL | non-hate | non-hate [END]\n")
                continue

            outputs = llm.generate(valid_prompts, sampling_params)

            idx = 0
            for prompt in batch_prompts:
                if prompt is None:
                    f_out.write("NULL | NULL | non-hate | non-hate [END]\n")
                else:
                    text = outputs[idx].outputs[0].text.strip()
                    if '[END]' in text:
                        text = text.split('[END]')[0].strip() + ' [END]'
                    else:
                        text = text + ' [END]'

                    if text.count('|') < 3:
                        f_out.write("NULL | NULL | non-hate | non-hate [END]\n")
                    else:
                        f_out.write(text + "\n")
                    idx += 1

    print(f"Inference complete. Output saved to {OUTPUT_TXT_PATH}")

if __name__ == "__main__":
    main()

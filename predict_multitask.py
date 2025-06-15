import json
import torch
import pandas as pd
from transformers import (
    BertTokenizerFast,
    BertPreTrainedModel,
    BertModel,
)
from tqdm.auto import tqdm
import numpy as np

# --- 1. 复现训练时的配置和模型结构 ---

# 确保这里的标签定义和 train_multitask3.py 完全一致
GROUP_LABELS_LIST = sorted(["Racism", "Region", "Sexism", "LGBTQ", "others", "non-hate"])
HATEFUL_LABELS_LIST = sorted(["hate", "non-hate"])

ID_TO_GROUP = {i: label for i, label in enumerate(GROUP_LABELS_LIST)}
ID_TO_HATEFUL = {i: label for i, label in enumerate(HATEFUL_LABELS_LIST)}

# 必须原封不动地复制训练时使用的模型类定义
class BertForMultiTaskHateDetection(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_group_labels = len(ID_TO_GROUP)
        self.num_hateful_labels = len(ID_TO_HATEFUL)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.target_qa_outputs = torch.nn.Linear(config.hidden_size, 2)
        self.argument_qa_outputs = torch.nn.Linear(config.hidden_size, 2)
        self.group_classifier = torch.nn.Linear(config.hidden_size, self.num_group_labels)
        self.hateful_classifier = torch.nn.Linear(config.hidden_size, self.num_hateful_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0]

        target_logits = self.target_qa_outputs(sequence_output)
        target_start_logits, target_end_logits = target_logits.split(1, dim=-1)
        target_start_logits = target_start_logits.squeeze(-1).contiguous()
        target_end_logits = target_end_logits.squeeze(-1).contiguous()

        argument_logits = self.argument_qa_outputs(sequence_output)
        argument_start_logits, argument_end_logits = argument_logits.split(1, dim=-1)
        argument_start_logits = argument_start_logits.squeeze(-1).contiguous()
        argument_end_logits = argument_end_logits.squeeze(-1).contiguous()

        pooled_output = self.dropout(cls_output)
        group_logits = self.group_classifier(pooled_output)
        hateful_logits = self.hateful_classifier(pooled_output)

        return {
            "target_start_logits": target_start_logits,
            "target_end_logits": target_end_logits,
            "argument_start_logits": argument_start_logits,
            "argument_end_logits": argument_end_logits,
            "group_logits": group_logits,
            "hateful_logits": hateful_logits,
        }

# --- 2. 预测主函数 ---
def predict():
    """
    使用训练好的多任务模型进行推理，并按要求格式化输出。
    """
    # --- 配置和加载 ---
    MODEL_PATH = "autodl-tmp/hate-speech-multitask-model/best"
    TEST_JSON_PATH = "autodl-tmp/test1.json"
    OUTPUT_TXT_PATH = "prediction.txt"
    MAX_LENGTH = 512
    
    print(f"正在从 {MODEL_PATH} 加载模型和分词器...")
    try:
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
        model = BertForMultiTaskHateDetection.from_pretrained(MODEL_PATH)
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
            original_content = row['content'].strip()
            if not original_content:
                continue

            masked_content = original_content
            extracted_tuples = []
            
            # 迭代抽取最多5个元组
            for _ in range(5): 
                inputs = tokenizer(
                    masked_content,
                    max_length=MAX_LENGTH,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    return_offsets_mapping=True
                )
                
                offset_mapping = inputs.pop("offset_mapping").squeeze(0).numpy()
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                # --- 解码分类结果 ---
                group_id = torch.argmax(outputs['group_logits'], dim=-1).item()
                hateful_id = torch.argmax(outputs['hateful_logits'], dim=-1).item()
                group_label = ID_TO_GROUP[group_id]
                hateful_label = ID_TO_HATEFUL[hateful_id]
                
                # --- 解码抽取结果 ---
                target_start_idx = torch.argmax(outputs['target_start_logits'], dim=-1).item()
                target_end_idx = torch.argmax(outputs['target_end_logits'], dim=-1).item()
                
                argument_start_idx = torch.argmax(outputs['argument_start_logits'], dim=-1).item()
                argument_end_idx = torch.argmax(outputs['argument_end_logits'], dim=-1).item()
                
                # --- 验证和文本还原 ---
                # 如果模型预测的开始位置在结束位置之后，或者指向[CLS]（索引为0），则认为没有有效抽取
                if target_start_idx > target_end_idx or target_start_idx == 0:
                    break
                
                # 从offset_mapping还原文本
                target_char_start = offset_mapping[target_start_idx][0]
                target_char_end = offset_mapping[target_end_idx][1]
                target_str = masked_content[target_char_start:target_char_end].strip()

                # 如果Target为空，也停止
                if not target_str:
                    break

                if argument_start_idx > argument_end_idx or argument_start_idx == 0:
                    argument_str = "NULL"
                else:
                    arg_char_start = offset_mapping[argument_start_idx][0]
                    arg_char_end = offset_mapping[argument_end_idx][1]
                    argument_str = masked_content[arg_char_start:arg_char_end].strip()
                    if not argument_str:
                        argument_str = "NULL"

                # 清理和组装
                target_clean = target_str.replace("|", "").strip()
                argument_clean = argument_str.replace("|", "").strip()

                # 如果是non-hate，强制将group设为non-hate
                if hateful_label == "non-hate":
                    group_label = "non-hate"

                quad = (target_clean, argument_clean, group_label, hateful_label)

                if quad in extracted_tuples: # 避免重复抽取
                    break
                
                extracted_tuples.append(quad)
                
                # 准备下一次迭代：屏蔽已抽出的Target
                masked_content = masked_content.replace(target_str, "█" * len(target_str), 1)

            # --- 格式化输出 ---
            if extracted_tuples:
                output_line = " [SEP] ".join([" | ".join(q) for q in extracted_tuples]) + " [END]\n"
                f_out.write(output_line)
            else:
                # 如果没有抽到任何内容，输出默认的 non-hate 元组
                f_out.write("NULL | NULL | non-hate | non-hate [END]\n")

    print(f"\n推理完成！预测结果已保存至 {OUTPUT_TXT_PATH}。")

if __name__ == "__main__":
    predict()
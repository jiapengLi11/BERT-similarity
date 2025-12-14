import os
import torch
from transformers import BertTokenizer
from model import build_model  # 导入模型构建函数
from predict import predict_similarity  # 导入预测函数
from config import *  # 导入配置（模型名、设备、最大长度等）


def main():
    # 1. 加载Tokenizer（需与训练时一致）
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    # 2. 构建模型并加载训练好的最优权重
    model = build_model(use_lora=USE_LORA)  # 传入LoRA配置

    # 根据是否使用LoRA采用不同的加载方式
    if USE_LORA:
        model.load_adapter(SAVE_MODEL_PATH.replace('.pt', ''), adapter_name="default")  # LoRA加载
    else:
        model.load_state_dict(torch.load(SAVE_MODEL_PATH))

    model.to(DEVICE)
    model.eval()

    # 3. 待预测的句子对（可替换为自己的句子）
    test_cases = [
        ("A cat is chasing a mouse.", "A feline is pursuing a rodent."),
        ("A woman is playing piano.", "A girl is playing a piano."),
        ("一位女孩在给另一位女孩测量身高.", "女孩给另一女孩量身高是多少."),
        ("一位老师在给一位学生讲故事.", "一位老师在给一位学生讲故事."),
    ]

    # 4. 批量预测并输出结果
    print("===== 相似度预测结果 =====")
    for idx, (sent1, sent2) in enumerate(test_cases, 1):
        score = predict_similarity(sent1, sent2, model, tokenizer)
        print(f"案例{idx}:")
        print(f"句子1: {sent1}")
        print(f"句子2: {sent2}")
        print(f"预测相似度分数: {score:.2f}\n")


if __name__ == "__main__":
    # 关闭无关警告（可选）
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    main()
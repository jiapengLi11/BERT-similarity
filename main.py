from preprocessor import build_dataloaders
from model import build_model, build_optimizer_scheduler
from train import train_model, test_model
from predict import predict_similarity
from config import *
# 在 main.py 中添加以下代码
from model import build_model, build_optimizer_scheduler, count_parameters

def main():
    # 1. 数据处理：构建DataLoader和Tokenizer
    print("========== Loading and Preprocessing Data ==========")
    train_loader, val_loader, test_loader, tokenizer = build_dataloaders()

    # 2. 构建模型、优化器、损失函数
    print("\n========== Building Model ==========")
    model = build_model(use_lora=USE_LORA)
    optimizer, scheduler, loss_fn = build_optimizer_scheduler(model, train_loader)

    # 3. 训练模型
    print("\n========== Starting Training ==========")
    train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn)

    # 4. 测试模型
    print("\n========== Starting Testing ==========")
    test_model(model, test_loader, loss_fn)

    # 5. 示例预测
    print("\n========== Example Prediction ==========")
    # 注意：LoRA模型保存和加载方式略有不同
    if USE_LORA:
        model.save_pretrained(SAVE_MODEL_PATH.replace('.pt', ''))  # LoRA保存为目录
    else:
        model.load_state_dict(torch.load(SAVE_MODEL_PATH))
        model.to(DEVICE)
    # 测试句子对
    sent1 = "A cat is chasing a mouse."
    sent2 = "A feline is pursuing a rodent."
    score = predict_similarity(sent1, sent2, model, tokenizer)
    print(f"Sentence 1: {sent1}")
    print(f"Sentence 2: {sent2}")
    print(f"Predicted Similarity Score: {score:.2f}")


if __name__ == "__main__":
    main()

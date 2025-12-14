import torch
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
import torch.optim as optim  # 替换 from transformers import AdamW
from config import *
from peft import LoraConfig, get_peft_model, TaskType  # 新增LoRA相关导入
# 统计模型参数
def count_parameters(model):
    """
    统计模型参数量
    返回: (总参数量, 可训练参数量)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# 构建BERT回归模型（支持LoRA微调）
def build_model(use_lora=False):
    # 加载BERT模型（回归任务）
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=NUM_LABELS
    )

    # 如果启用LoRA微调
    if use_lora:
        # 配置LoRA参数
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,#序列分类/回归任务
            inference_mode=False,#推理模式
            r=8,  # LoRA秩（秩越小，参数量越少8/16）
            lora_alpha=16,#LoRA缩放因子（r的4倍）
            lora_dropout=0.1,#LoRA的dropout
            target_modules=["query", "value"],  #仅需微调attention的query和value模块加Lora
            bias="none",#不训练LoRA的偏置
            modules_to_save=["classifier"]#保存回归头的训练（原有分类头）
        )
        model = get_peft_model(model, peft_config)#加载LoRA模型
        print("LoRA微调已启用")

        # 在这里就统计参数
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"参数效率: {trainable_params / total_params * 100:.2f}%")
    else:
        print("全参数微调模式")

    # 移到指定设备
    model.to(DEVICE)
    return model


# 构建优化器和学习率调度器
def build_optimizer_scheduler(model, train_loader):
    # 使用PyTorch原生AdamW，消除弃用警告
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        eps=EPS,
        weight_decay=0.01  # 权重衰减，正则化
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    loss_fn = torch.nn.MSELoss()
    return optimizer, scheduler, loss_fn

import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from config import *
from torch.cuda.amp import autocast, GradScaler
# 在训练代码开头添加
import torch
print(f"当前使用GPU：{torch.cuda.get_device_name(0)}")  # 输出GTX 16xx
print(f"初始GPU显存占用：{torch.cuda.memory_allocated(0)/1024**3:.2f}GB")

def validate(model, val_loader, loss_fn, scaler=None):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            # 使用混合精度训练
            if USE_AMP and scaler is not None:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    logits = outputs.logits.view(-1)
                    labels = labels.view(-1)
                    loss = loss_fn(logits, labels)
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                # 关键修改：用view(-1)替代squeeze()，强制转为1维张量（避免0维）
                logits = outputs.logits.view(-1)  # 形状：[batch_size]
                # 确保labels也是1维（与logits匹配）
                labels = labels.view(-1)
                loss = loss_fn(logits, labels)

            val_loss += loss.item()
            # 收集预测值和标签（1维数组，可迭代）
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    pearson_corr = pearsonr(all_preds, all_labels)[0]
    spearman_corr = spearmanr(all_preds, all_labels)[0]
    return avg_val_loss, pearson_corr, spearman_corr


# 训练函数
def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn):
    best_pearson = 0.0
    patience = 2
    patience_counter = 0
    # 添加混合精度训练支持
    scaler = GradScaler() if USE_AMP else None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS}")

        # 梯度累积需要在每个epoch开始时清零梯度
        optimizer.zero_grad()

        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # 混合精度训练
            if USE_AMP and scaler is not None:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    logits = outputs.logits.view(-1)
                    labels = labels.view(-1)
                    # 计算损失并除以累积步数
                    loss = loss_fn(logits, labels) / GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                # 关键修改：用view(-1)替代squeeze()，保证1维
                logits = outputs.logits.view(-1)
                labels = labels.view(-1)  # 匹配维度
                # 计算损失并除以累积步数
                loss = loss_fn(logits, labels) / GRADIENT_ACCUMULATION_STEPS
                loss.backward()

            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            # 梯度累积：只有在累积步数完成时才更新参数
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if USE_AMP and scaler is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # 清零梯度并更新学习率
                optimizer.zero_grad()
                scheduler.step()

            progress_bar.set_postfix({"train_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})

        # 处理最后一个批次（如果总步数不能被累积步数整除）
        if (len(train_loader)) % GRADIENT_ACCUMULATION_STEPS != 0:
            if USE_AMP and scaler is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

        # 计算训练集平均损失
        avg_train_loss = train_loss / len(train_loader)

        # 验证集评估
        avg_val_loss, pearson_corr, spearman_corr = validate(model, val_loader, loss_fn, scaler if USE_AMP else None)

        # 打印日志
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Pearson Corr: {pearson_corr:.4f} | Val Spearman Corr: {spearman_corr:.4f}")

        # 保存最优模型
        if pearson_corr > best_pearson:
            best_pearson = pearson_corr
            if USE_LORA:
                # LoRA模式下保存模型
                model.save_pretrained(SAVE_MODEL_PATH.replace('.pt', ''))
            else:
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"Best model saved (Pearson: {best_pearson:.4f})")


# 测试函数
def test_model(model, test_loader, loss_fn):
    # 加载最优模型
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.to(DEVICE)
    # 测试集评估
    test_loss, test_pearson, test_spearman = validate(model, test_loader, loss_fn,
                                                      GradScaler() if USE_AMP else None)
    print("\n==================== Test Set Results ====================")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Pearson Corr: {test_pearson:.4f} | Test Spearman Corr: {test_spearman:.4f}")
    return test_loss, test_pearson, test_spearman

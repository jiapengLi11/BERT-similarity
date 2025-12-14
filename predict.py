import torch
from transformers import BertTokenizer
from config import *


def predict_similarity(sentence1, sentence2, model, tokenizer):
    model.eval()
    encodings = tokenizer(
        [sentence1], [sentence2],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)
    token_type_ids = encodings["token_type_ids"].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # 关键：用view(-1)保证1维，再取第一个值
        score = outputs.logits.view(-1).cpu().numpy()[0]
        score = max(0.0, min(5.0, score))
    return score
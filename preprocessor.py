import pandas as pd
import torch
from transformers import BertTokenizer
from config import *


# 加载已划分好的无表头数据集（核心修改）
def load_and_split_data():
    # 读取无表头数据：header=None 表示无列名；sep='\t' 适配STS-B默认的制表符分隔（若为逗号则改 sep=','）
    # 若数据是空格分隔，可尝试 sep='\s+'；若为固定宽度，用 pd.read_fwf
    train_df = pd.read_csv(TRAIN_DATA_PATH, header=None, sep='\t')
    val_df = pd.read_csv(VAL_DATA_PATH, header=None, sep='\t')
    test_df = pd.read_csv(TEST_DATA_PATH, header=None, sep='\t')

    # 按列索引提取数据：第0列=句子1，第1列=句子2，第2列=相似度分数
    train_data = (
        train_df.iloc[:, 0].tolist(),  # 第一列（索引0）：句子1
        train_df.iloc[:, 1].tolist(),  # 第二列（索引1）：句子2
        train_df.iloc[:, 2].tolist()  # 第三列（索引2）：分数
    )
    val_data = (
        val_df.iloc[:, 0].tolist(),
        val_df.iloc[:, 1].tolist(),
        val_df.iloc[:, 2].tolist()
    )
    test_data = (
        test_df.iloc[:, 0].tolist(),
        test_df.iloc[:, 1].tolist(),
        test_df.iloc[:, 2].tolist()
    )
    return train_data, val_data, test_data


# 文本编码函数（不变）
def encode_texts(texts1, texts2, tokenizer):
    encodings = tokenizer(
        texts1, texts2,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    return encodings


# Dataset类（不变）
class STSBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# 构建DataLoader（不变）
def build_dataloaders():
    # 加载tokenizer（若下载慢，可手动下载后指定本地路径）
    #tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)#本地加载
    # 加载数据
    train_data, val_data, test_data = load_and_split_data()
    train_texts1, train_texts2, train_labels = train_data
    val_texts1, val_texts2, val_labels = val_data
    test_texts1, test_texts2, test_labels = test_data

    # 编码文本
    train_encodings = encode_texts(train_texts1, train_texts2, tokenizer)
    val_encodings = encode_texts(val_texts1, val_texts2, tokenizer)
    test_encodings = encode_texts(test_texts1, test_texts2, tokenizer)

    # 转换标签为张量（浮点型，适配回归任务）
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # 构建Dataset和DataLoader
    train_dataset = STSBDataset(train_encodings, train_labels)
    val_dataset = STSBDataset(val_encodings, val_labels)
    test_dataset = STSBDataset(test_encodings, test_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer
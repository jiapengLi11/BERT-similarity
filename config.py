import torch

# 模型相关配置
#MODEL_NAME = "bert-base-chinese"  # 英文：bert-base-uncased | 中文：bert-base-chinese
MODEL_PATH = './models/bert-base-chinese'#本地加载模型
NUM_LABELS = 1  # 回归任务，输出1个连续值
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_LORA = True  # 是否使用LoRA微调

# 数据相关配置（修改为已分好的数据集路径）
TRAIN_DATA_PATH = "./STS-B/STS-B.train.data"  # 训练集路径
VAL_DATA_PATH = "./STS-B/STS-B.valid.data"  # 验证集路径
TEST_DATA_PATH = "./STS-B/STS-B.test.data"  # 测试集路径
MAX_LENGTH = 256  # 文本最大长度
RANDOM_STATE = 42  # 随机种子（保证可复现）

# 训练相关配置
BATCH_SIZE = 8
# 配置：batch_size=8（原16），梯度累积步数=2
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 12
LEARNING_RATE = 5e-5#lora可训练参数量减少，可适当增大学习率5e-5
EPS = 1e-8  # AdamW优化器epsilon
WARMUP_STEPS = 0  # 学习率预热步数
# 混合精度训练配置
USE_AMP = False  # 是否使用混合精度训练

# 路径配置
SAVE_MODEL_PATH = "best-sts-b-model.pt"  # 最优模型保存路径

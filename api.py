from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer
from model import build_model
from predict import predict_similarity
from config import *
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
#启动 API 服务
#uvicorn api:app --host 0.0.0.0 --port 8000 --reload
#直接打开浏览器访问 http://localhost:8000



# 初始化FastAPI
app = FastAPI(title="句子相似度计算API")
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 定义请求体格式
class SentencePair(BaseModel):
    sentence1: str
    sentence2: str


# 加载模型和Tokenizer（启动时执行一次）
@app.on_event("startup")
def load_model_and_tokenizer():
    global model, tokenizer
    # 加载Tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    # 构建模型
    model = build_model(use_lora=USE_LORA)
    # 加载权重
    if USE_LORA:
        model.load_adapter(SAVE_MODEL_PATH.replace('.pt', ''), adapter_name="default")
    else:
        model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("模型加载完成，准备接收请求")


# 相似度预测接口
@app.post("/predict", response_model=dict)
def predict(pair: SentencePair):
    try:
        if not pair.sentence1 or not pair.sentence2:
            raise HTTPException(status_code=400, detail="句子不能为空")

        score = predict_similarity(
            pair.sentence1,
            pair.sentence2,
            model,
            tokenizer
        )
        return {
            "sentence1": pair.sentence1,
            "sentence2": pair.sentence2,
            "similarity_score": round(float(score), 2),
            "message": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 健康检查接口
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "model" in globals()}
# 前端页面入口
@app.get("/")
def read_index():
    return FileResponse("static/index.html")
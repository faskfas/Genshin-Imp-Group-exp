import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import load_dataset, DatasetDict
import accelerate
from tqdm import tqdm

# -------------------------- 1. 配置参数（适配ChnSentiCorp） --------------------------
class Config:
    # 基础配置
    model_name = "bert-base-chinese"  # 中文BERT预训练模型
    num_labels = 2  # ChnSentiCorp是二分类（0=负面，1=正面）
    max_length = 128  # 该数据集文本长度多在128以内
    batch_size = 32  # 适配常规GPU显存（可根据显存调整为16/8）
    epochs = 1
    learning_rate = 2e-5  # BERT微调经典学习率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./chnsenticorp_bert_output"  # 模型输出目录
    seed = 42  # 固定随机种子保证可复现
    train_sample_ratio = 0.1

# 设置随机种子
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)

class TqdmProgressCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.total_steps = 0
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # 初始化进度条：总步数 = epoch数 * 每个epoch的步数
        self.total_steps = state.max_steps
        self.pbar = tqdm(
            total=self.total_steps,
            desc="训练进度",
            unit="step",
            ncols=100,  # 进度条宽度
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

    def on_step_end(self, args, state, control, **kwargs):
        # 每步更新进度条
        self.current_step += 1
        self.pbar.update(1)
        # 显示实时损失（可选）
        if state.log_history and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                self.pbar.set_postfix({"训练损失": f"{last_log['loss']:.4f}"})

    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束关闭进度条
        self.pbar.close()
        print(f"\n训练完成！总步数：{self.total_steps}")

    def on_evaluate(self, args, state, control, **kwargs):
        # 评估时显示提示
        tqdm.write("\n开始评估验证集...")

# -------------------------- 2. 加载ChnSentiCorp数据集 --------------------------
dataset = load_dataset("lansinuote/ChnSentiCorp")

# 数据集结构
print("数据集基本信息：")
print(f"训练集样本数：{len(dataset['train'])}")
print(f"验证集样本数：{len(dataset['validation'])}")
print(f"测试集样本数：{len(dataset['test'])}")
print(f"样本示例：{dataset['train'][0]}")

# -------------------------- 3. 数据预处理 --------------------------
# 加载BERT中文分词器
tokenizer = BertTokenizer.from_pretrained(Config.model_name)

def preprocess_function(examples):
    """
    文本分词处理：
    - 截断过长文本
    - 填充到max_length
    - 返回PyTorch张量
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=Config.max_length,
        return_tensors="pt"
    )

# 批量预处理数据集
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 格式化数据集
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 指定模型输入列
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# -------------------------- 4. 加载预训练BERT模型 --------------------------
model = BertForSequenceClassification.from_pretrained(
    Config.model_name,
    num_labels=Config.num_labels,  # 二分类任务
    ignore_mismatched_sizes=True  # 兼容部分预训练模型的权重维度
).to(Config.device)

# -------------------------- 5. 定义训练参数 --------------------------
training_args = TrainingArguments(
    output_dir=Config.output_dir,
    num_train_epochs=Config.epochs, 
    per_device_train_batch_size=Config.batch_size,
    per_device_eval_batch_size=Config.batch_size,
    learning_rate=Config.learning_rate,
    # 评估与保存策略
    evaluation_strategy="epoch",  # 每个epoch评估一次
    save_strategy="epoch",        # 每个epoch保存一次模型
    load_best_model_at_end=True,  # 训练结束加载最优模型
    metric_for_best_model="accuracy",
    # 防止过拟合
    weight_decay=0.01,  # 权重衰减
    # 日志与效率
    logging_dir=f"{Config.output_dir}/logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),  # GPU开启混合精度训练
    remove_unused_columns=False,     # 保留数据集所有列
)

# -------------------------- 6. 定义评估指标--------------------------
def compute_metrics(eval_pred):
    """计算准确率、精确率、召回率、F1"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # 取logits最大值为预测标签
    
    # 计算核心指标
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "positive_precision": report["1"]["precision"],  # 正面情感精确率
        "positive_recall": report["1"]["recall"]         # 正面情感召回率
    }

# -------------------------- 7. 训练模型 --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    callbacks=[TqdmProgressCallback(),EarlyStoppingCallback(early_stopping_patience=2)]  # 早停（2个epoch无提升则停止）
)

# 开始训练
print("\n========== 开始训练BERT模型 ==========")
trainer.train()

# -------------------------- 8. 评估模型 --------------------------
print("\n========== 验证集评估结果 ==========")
val_results = trainer.evaluate(tokenized_datasets["validation"])
for key, value in val_results.items():
    if key in ["accuracy", "precision", "recall", "f1", "positive_precision", "positive_recall"]:
        print(f"{key}: {value:.4f}")

print("\n========== 测试集评估结果 ==========")
test_results = trainer.evaluate(tokenized_datasets["test"])
for key, value in test_results.items():
    if key in ["accuracy", "precision", "recall", "f1", "positive_precision", "positive_recall"]:
        print(f"{key}: {value:.4f}")

# -------------------------- 9. 模型预测（单文本/批量文本） --------------------------
def predict_sentiment(texts, model, tokenizer, device):
    """
    情感预测函数：
    - texts: 文本列表（支持单文本/批量文本）
    - 返回：预测标签（0/1）+ 情感标签（负面/正面）
    """
    # 统一转为列表处理
    if isinstance(texts, str):
        texts = [texts]
    
    # 文本预处理
    inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=Config.max_length,
        return_tensors="pt"
    ).to(device)
    
    # 模型推理（关闭梯度计算加速）
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
    
    # 标签映射
    label_map = {0: "负面", 1: "正面"}
    results = [{"text": text, "pred_label": int(label), "sentiment": label_map[label]} 
               for text, label in zip(texts, pred_labels)]
    
    return results

# 测试预测
test_texts = [
    "这家酒店的服务特别好，房间也很干净，下次还来！",
    "快递速度太慢了，商品还破损了，非常不满意",
    "电影剧情一般，但演员演技还不错",
    "这个手机用了一周就卡得不行，千万别买"
]

print("\n========== 预测示例 ==========")
predictions = predict_sentiment(test_texts, model, tokenizer, Config.device)
for pred in predictions:
    print(f"文本：{pred['text']}")
    print(f"预测标签：{pred['pred_label']} | 情感：{pred['sentiment']}\n")

# -------------------------- 10. 保存模型与分词器 --------------------------
model_save_path = f"{Config.output_dir}/best_bert_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\n最优模型已保存至：{model_save_path}")
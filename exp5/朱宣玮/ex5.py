import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import os

# 设置网络代理，以直接从Hugging Face上下载数据集
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_LENGTH = 128  # 句子最大长度

# 全连接层模型
class FCModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=1):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 工具函数
def binary_accuracy(predictions, labels):
    """计算二分类准确率"""
    rounded_preds = torch.round(predictions)
    correct = (rounded_preds == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def preprocess_function(examples, tokenizer):
    """预处理函数：将文本转换为模型输入"""
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

def evaluate(model, bert_model, eval_loader, criterion, device):
    """在验证集上评估模型性能"""
    model.eval()  # 切换到评估模式
    bert_model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    # 关闭梯度计算，节省计算资源和时间
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].float().to(device)

            # 前向传播
            bert_output = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            pooler_output = bert_output.pooler_output
            predictions = model(pooler_output).squeeze()

            # 计算损失和准确率
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            # 累计指标
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size

    # 计算平均指标
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc

# 主训练流程
def main():
    # 1. 自动载入MRPC数据集
    print("正在载入MRPC数据集...")
    dataset = load_dataset("glue", "mrpc")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"] # 加载验证集
    print(f"数据集载入完成，训练集: {len(train_dataset)} 个样本, 验证集: {len(eval_dataset)} 个样本")

    # 2. 设置运行设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 3. 加载BERT模型和Tokenizer
    print("正在加载BERT模型...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.to(device)
    print("BERT模型加载完成")

    # 4. 创建全连接层模型
    print("正在创建全连接层模型...")
    model = FCModel()
    model.to(device)
    print("全连接层模型创建完成")

    # 5. 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()  # 二分类交叉熵损失

    # 6. 预处理数据集
    print("正在预处理训练集...")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True
    )
    tokenized_train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"]
    )

    print("正在预处理验证集...")
    tokenized_eval_dataset = eval_dataset.map( # 预处理验证集
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True
    )
    tokenized_eval_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"]
    )
    print("数据集预处理完成")

    # 7. 创建DataLoader
    train_loader = DataLoader(
        tokenized_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    eval_loader = DataLoader( # 创建验证集DataLoader
        tokenized_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False # 验证时不需要打乱
    )

    # 8. 训练模型
    print("开始训练...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")
        
        # --- 训练阶段 ---
        model.train()
        bert_model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].float().to(device)
            
            optimizer.zero_grad()
            bert_optimizer.zero_grad()
            
            bert_output = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            pooler_output = bert_output.pooler_output
            predictions = model(pooler_output).squeeze()
            
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            
            loss.backward()
            optimizer.step()
            bert_optimizer.step()
            
            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_acc += acc.item() * batch_size
            total_samples += batch_size
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
        
        avg_train_loss = epoch_loss / total_samples
        avg_train_acc = epoch_acc / total_samples
        print(f"训练集 - 平均损失: {avg_train_loss:.4f}, 平均准确率: {avg_train_acc:.4f}")

        # --- 验证阶段 ---
        print("正在进行验证...")
        avg_eval_loss, avg_eval_acc = evaluate(model, bert_model, eval_loader, criterion, device)
        print(f"验证集 - 平均损失: {avg_eval_loss:.4f}, 平均准确率: {avg_eval_acc:.4f}")

    print("\n训练完成！")

# -------------------------- 运行入口 --------------------------
if __name__ == "__main__":
    main()
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int = 42):
    """保证实验可复现."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """自动选择设备: CUDA > MPS(Apple) > CPU."""
    if torch.cuda.is_available():
        print("使用设备: CUDA")
        return torch.device("cuda")
    # 某些 torch 版本没有 mps 属性，这里做一下兼容
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("使用设备: MPS (Apple 芯片)")
        return torch.device("mps")
    print("使用设备: CPU")
    return torch.device("cpu")


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    labels: torch.Tensor


def collate_fn(features: List[Dict], pad_token_id: int) -> Batch:
    """自定义 collate_fn，做动态 padding."""
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    token_type_ids = [f["token_type_ids"] for f in features]
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

    max_len = max(len(x) for x in input_ids)

    def pad(seqs, pad_value):
        return torch.stack(
            [
                torch.tensor(
                    list(seq) + [pad_value] * (max_len - len(seq)), dtype=torch.long
                )
                for seq in seqs
            ]
        )

    input_ids = pad(input_ids, pad_token_id)
    attention_mask = pad(attention_mask, 0)
    token_type_ids = pad(token_type_ids, 0)

    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels,
    )


def preprocess_function(examples, tokenizer, max_length: int = 128):
    """对 MRPC 的句子对进行分词编码."""
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=max_length,
    )


def create_dataloaders(tokenized_datasets, tokenizer, batch_size: int = 16):
    """根据 tokenized 的数据集创建 DataLoader."""
    pad_token_id = tokenizer.pad_token_id

    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    )

    eval_loader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    )

    test_loader = DataLoader(
        tokenized_datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    )

    return train_loader, eval_loader, test_loader


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch_idx: int,
):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1} 训练中")

    for batch in progress_bar:
        batch = Batch(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            token_type_ids=batch.token_type_ids.to(device),
            labels=batch.labels.to(device),
        )

        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            token_type_ids=batch.token_type_ids,
            labels=batch.labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        # 可选: 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device, desc: str = "验证集评估"):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=desc)
        for batch in progress_bar:
            batch = Batch(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                token_type_ids=batch.token_type_ids.to(device),
                labels=batch.labels.to(device),
            )

            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                token_type_ids=batch.token_type_ids,
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_labels.extend(batch.labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1


def main():
    set_seed(42)
    device = get_device()

    # 1. 加载 MRPC 数据集
    print("正在加载 GLUE MRPC 数据集...")
    raw_datasets = load_dataset("glue", "mrpc")

    # 2. 加载 tokenizer 和 BERT 分类模型
    print("正在加载 bert-base-uncased 模型...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.to(device)

    # 3. 文本预处理与编码
    print("正在对数据集进行分词编码...")
    tokenized_datasets = raw_datasets.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        desc="Tokenization",
    )

    # 只保留模型需要的字段
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["idx", "sentence1", "sentence2"]
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # 4. 构建 DataLoader
    train_loader, eval_loader, test_loader = create_dataloaders(
        tokenized_datasets, tokenizer, batch_size=16
    )

    # 5. 设置优化器和学习率调度
    num_epochs = 3
    total_steps = len(train_loader) * num_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 6. 训练循环
    print("开始训练 BERT 模型（MRPC 任务）...")
    best_eval_f1 = 0.0
    best_state_dict = None

    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        print(f"Epoch {epoch+1}/{num_epochs} 平均训练损失: {avg_train_loss:.4f}")

        eval_acc, eval_f1 = evaluate(model, eval_loader, device, desc="验证集评估")
        print(
            f"验证集准确率: {eval_acc:.4f}, 验证集 F1 分数: {eval_f1:.4f}"
        )

        # 记录最佳模型参数（按 F1）
        if eval_f1 > best_eval_f1:
            best_eval_f1 = eval_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 如果有更好的验证集模型，将其加载回来
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    # 7. 在测试集上做最终评估
    test_acc, test_f1 = evaluate(model, test_loader, device, desc="测试集评估")
    print("在测试集上的最终结果：")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测试集 F1 分数: {test_f1:.4f}")

    # 8. 保存微调后的模型和 tokenizer
    save_dir = "bert_mrpc_finetuned"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"微调后的模型已保存到: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()

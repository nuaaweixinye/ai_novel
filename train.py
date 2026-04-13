import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset


class ChineseNovelDataset(Dataset):
    """中文小说数据集类"""
    def __init__(self, tokenizer, file_path, block_size=128):
        self.examples = []
        
        print(f"正在加载训练数据: {file_path}")
        with open(file_path, encoding='utf-8') as f:
            text = f.read()
        
        # 获取模型最大长度限制
        max_length = min(tokenizer.model_max_length, 1024)  # 限制最大长度为1024
        
        # 分割文本为较小的块，确保不超过模型最大长度
        # 将整个文本按字符分割成较小的块
        text_chunks = []
        for i in range(0, len(text), 500):  # 每500个字符为一组
            chunk = text[i:i + 500]
            if chunk.strip():  # 只添加非空块
                text_chunks.append(chunk)
        
        # 处理每个文本块
        for chunk in text_chunks:
            if len(chunk) == 0:
                continue
                
            # 编码文本块
            tokenized_chunk = tokenizer.encode(chunk, add_special_tokens=True)
            
            # 如果编码后的长度超过限制，只取前面的部分
            if len(tokenized_chunk) > max_length:
                tokenized_chunk = tokenized_chunk[:max_length]
                
            # 将文本块按block_size分割
            for i in range(0, len(tokenized_chunk) - block_size + 1, block_size // 2):
                sub_chunk = tokenized_chunk[i:i + block_size]
                if len(sub_chunk) == block_size:  # 确保块大小一致
                    self.examples.append(torch.tensor(sub_chunk))
            
            # 如果最后剩余部分不足block_size但大于等于block_size的一半，也作为一个样本
            remaining = len(tokenized_chunk) % block_size
            if remaining >= block_size // 2 and len(tokenized_chunk) >= block_size:
                start_idx = ((len(tokenized_chunk) // block_size) * block_size) - block_size // 2
                if start_idx >= 0:
                    sub_chunk = tokenized_chunk[start_idx:start_idx + block_size]
                    if len(sub_chunk) == block_size:
                        self.examples.append(torch.tensor(sub_chunk))
        
        print(f"总共创建了 {len(self.examples)} 个训练样本")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {"input_ids": self.examples[i]}


def train_novel_model():
    """训练中文小说模型"""
    print("开始训练中文小说模型...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载中文预训练模型
    # 尝试使用中文GPT模型
    model_names = [
        "uer/gpt2-chinese-cluecorpussmall",  # 较小的中文GPT模型
        "ckiplab/gpt2-base-chinese",        # 中文GPT模型
        "thaedalian/gpt-j-6B-chinese"       # 如果有更大内存可用
    ]
    
    model = None
    tokenizer = None
    
    for model_name in model_names:
        try:
            print(f"尝试加载模型: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print(f"成功加载模型: {model_name}")
            break
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {e}")
            continue
    
    # 如果都没成功，使用基础GPT-2
    if model is None:
        print("使用基础GPT-2模型...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # 添加中文支持
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    train_file = "data/train.txt"
    if not os.path.exists(train_file):
        print(f"错误: 找不到训练文件 {train_file}")
        return
        
    dataset = ChineseNovelDataset(tokenizer, train_file, block_size=128)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言建模而非掩码语言建模
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./novel_model",
        num_train_epochs=3,  # 减少训练轮数以加快训练
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-5,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),  # GPU上启用混合精度
        dataloader_num_workers=0,  # Windows兼容性
        remove_unused_columns=False,
        report_to=[],  # 禁用wandb等报告工具
        seed=42,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    print("开始训练过程...")
    print(f"训练样本数量: {len(dataset)}")
    print(f"训练轮数: {training_args.num_train_epochs}")
    print(f"批次大小: {training_args.per_device_train_batch_size}")
    print(f"学习率: {training_args.learning_rate}")
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    print("训练完成，正在保存模型...")
    model.save_pretrained("./novel_model")
    tokenizer.save_pretrained("./novel_model")
    
    print("模型已保存到 ./novel_model 目录")
    
    return trainer


def main():
    # 直接执行训练功能
    train_novel_model()


if __name__ == "__main__":
    main()
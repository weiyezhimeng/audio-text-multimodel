from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
from load_data import load_data
import json
import numpy as np
import librosa
import gc
import math
from tqdm import tqdm
import os

# 将语音用whisper转为向量（embedding），输入LLM
# 将对应的文本回答的Embedding衔接到语音Embedding后面
# 计算loss损失函数并更新projecter的参数

class multi_model(nn.Module): 
    def __init__(self, model_lora, model_whisper_encoder, mode_train):
        super(multi_model, self).__init__()
        self.mode_train = mode_train
        self.lora_llama = model_lora
        self.whisper_encoder = model_whisper_encoder
        
        # 加入卷积层来缩小维度
        self.conv = nn.Conv1d(in_channels=1280, out_channels=1280, kernel_size=25, stride=25)
        
        # 用于进一步处理特征的全连接层
        self.projecter = nn.Sequential(
            nn.Linear(1280, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
        )

        # 冻结语音特征提取模块的信息
        for name, param in self.whisper_encoder.named_parameters():
            param.requires_grad = False
    

    def forward(self, audio_input_features, label_embeddings, input_embedding_user_prefix, input_embedding_assistant_prefix, attention_mask_label_ids, attention_mask_user_prefix, attention_mask_assistant_prefix):
        # 获取whisper encoder的特征
        whisper_encoder_feature = self.whisper_encoder(audio_input_features)["last_hidden_state"]
        
        # 计算需要填充的大小，确保序列长度可以被步幅整除
        sequence_length = whisper_encoder_feature.size(1)
        stride = 25
        padding_size = (math.ceil(sequence_length / stride) * stride) - sequence_length
        
        # 对输入进行填充，确保序列长度可以被步幅整除
        if padding_size > 0:
            whisper_encoder_feature = nn.functional.pad(whisper_encoder_feature, (0, padding_size), "constant", 0)
            print(f"输入长度 {sequence_length} 被填充到 {whisper_encoder_feature.size(1)}")
        
        # 将特征的形状从 [batch_size, sequence_length, feature_dim] 转换为 [batch_size, feature_dim, sequence_length]
        whisper_encoder_feature = whisper_encoder_feature.permute(0, 2, 1)
        
        # 使用卷积层压缩序列的长度
        compressed_feature = self.conv(whisper_encoder_feature)  # 输出形状: [batch_size, 1280, compressed_sequence_length]
        
        # 还原形状回 [batch_size, compressed_sequence_length, feature_dim]
        compressed_feature = compressed_feature.permute(0, 2, 1)

        # 使用全连接层处理特征
        audio_to_text = self.projecter(compressed_feature)
        
        if not self.mode_train:
            return torch.cat([input_embedding_user_prefix, audio_to_text, input_embedding_assistant_prefix], dim=1)
        # 合并输入（文本与音频特征）
        final_input = torch.cat([input_embedding_user_prefix, audio_to_text, input_embedding_assistant_prefix, label_embeddings], dim=1)
        attention_mask_audio = torch.ones(audio_to_text.shape[0], audio_to_text.shape[1], dtype=torch.long).to(final_input.device)
        attention_mask_final = torch.cat([attention_mask_user_prefix, attention_mask_audio, attention_mask_assistant_prefix, attention_mask_label_ids], dim=1)
        final_output = self.lora_llama(inputs_embeds=final_input, attention_mask=attention_mask_final).logits
        return final_output

# 保存方法
def save_model(model_final, model_lora, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 保存LoRA权重
    lora_dir = os.path.join(save_dir, "lora_weights")
    model_lora.save_pretrained(lora_dir)
    print(f"LoRA weights saved to {lora_dir}")

    # 保存multi_model中的可训练参数
    trainable_state_dict = {k: v for k, v in model_final.state_dict().items() if v.requires_grad}
    model_state_path = os.path.join(save_dir, "trainable_multi_model.pth")
    torch.save(trainable_state_dict, model_state_path)
    print(f"Trainable parts of multi_model saved to {model_state_path}")

if __name__ == "__main__":
    # 加载模型和相对应的tokenizer
    device = "cuda:0"
    path_llama = "Llama-2-7b-chat-hf"
    model_llama = AutoModelForCausalLM.from_pretrained(path_llama, torch_dtype=torch.float16).to(device)
    llama_embedding = model_llama.get_input_embeddings()
    for name, param in llama_embedding.named_parameters():
        param.requires_grad = False
    
    tokenizer_llama = AutoTokenizer.from_pretrained(path_llama, padding_side='right')
    tokenizer_llama.pad_token = tokenizer_llama.unk_token

    path_whisper = "whisper-large-v3-turbo"
    model_whisper_encoder = AutoModelForSpeechSeq2Seq.from_pretrained(
        path_whisper, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
    ).model.encoder.to(device)
    processor_whisper = AutoProcessor.from_pretrained(path_whisper)

    # lora初始化
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 将Llama模型套上lora的壳子
    model_lora = get_peft_model(model_llama, config)
    model_lora.print_trainable_parameters()
    
    # 初始化最终的模型
    model_final = multi_model(model_lora, model_whisper_encoder, mode_train=True).to(torch.float16).to(device)
    model_final.train()

    # 预处理数据集
    with open('./dataset/alpaca_data-0-3252-中文-已完成.json', 'r') as file:
        data = json.load(file)
    for item in tqdm(data):
        audio_file_path = item["path"]
        array, sr = librosa.load(audio_file_path, sr=16000)
        input_features = processor_whisper(array, sampling_rate=16000, return_tensors="pt").input_features.to(torch.float16)
        item["input_features"] = input_features.squeeze(0)

    batch = 1
    train_dataloader = load_data(data, batch)
    loss_fn = torch.nn.CrossEntropyLoss()
    vocab_size = tokenizer_llama.vocab_size
    optimizer = torch.optim.AdamW(model_final.parameters(), lr=1e-5, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    
    loss_all = []
    num_batches = len(train_dataloader)
    tenth_epoch_batches = num_batches // 10  # 计算十分之一个epoch的batch数量
    # batch accumulation parameter
    accum_iter = 32
    for epoch in range(10):
        print("Epoch:", epoch)
        for idx, (input_features, text_for_label) in enumerate(train_dataloader):
            
            # 训练的目标是label_ids
            label_ids = tokenizer_llama(text_for_label, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to(device)
            attention_mask_label_ids = tokenizer_llama(text_for_label, return_tensors="pt", padding=True, add_special_tokens=False)["attention_mask"].to(device)
            label_embedding = llama_embedding(label_ids).detach()
            # print(label_ids)
            
            user_prefix = "<s> [INST]"
            input_ids_user_prefix = tokenizer_llama(user_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            attention_mask_user_prefix = tokenizer_llama(user_prefix, return_tensors="pt", padding=True, add_special_tokens=False)["attention_mask"].repeat(batch, 1).to(device)
            input_embedding_user_prefix = llama_embedding(input_ids_user_prefix).repeat(batch, 1, 1).detach()

            assistant_prefix = "[/INST]"
            input_ids_assistant_prefix = tokenizer_llama(assistant_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            attention_mask_assistant_prefix = tokenizer_llama(assistant_prefix, return_tensors="pt", padding=True, add_special_tokens=False)["attention_mask"].repeat(batch, 1).to(device)
            input_embedding_assistant_prefix = llama_embedding(input_ids_assistant_prefix).repeat(batch, 1, 1).detach()
            
            input_features = input_features.to(device).detach()
            final_output = model_final(input_features, label_embedding, input_embedding_user_prefix, input_embedding_assistant_prefix, attention_mask_label_ids, attention_mask_user_prefix, attention_mask_assistant_prefix)[:, -label_ids.shape[-1]-1:-1, :]
            
            # mask掉label_ids中的padding部分
            label_ids[label_ids == 0] = -100
            loss = loss_fn(final_output.reshape(-1, vocab_size), label_ids.reshape(-1))
            loss = loss / accum_iter
            
            if (idx + 1) % tenth_epoch_batches == 0:
                mini_epoch = (idx + 1) / tenth_epoch_batches
                print(f"Epoch: {mini_epoch}  Loss: {loss}")
                loss_all.append(loss.item())
            if loss == float('nan'):
                continue
            loss.backward()
            
            if ((idx + 1) % accum_iter == 0) or (idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            del label_ids, label_embedding, input_ids_user_prefix, input_embedding_user_prefix, input_ids_assistant_prefix, input_embedding_assistant_prefix, input_features, final_output, loss; gc.collect(); torch.cuda.empty_cache()
        scheduler.step()
    
    # 保存模型
    save_dir = "./saved_model"
    save_model(model_final, model_lora, save_dir)
    # 将loss_all保存为JSON文件
    with open('loss_all.json', 'w') as f:
        json.dump(loss_all, f, indent = 4)

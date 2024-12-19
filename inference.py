from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import torch.nn as nn
import librosa
import math
from peft import PeftModel

class multi_model(nn.Module): 
    def __init__(self, model_lora, model_whisper_encoder, mode_train):
        super(multi_model, self).__init__()
        self.mode_train = mode_train
        self.lora_llama = model_lora
        self.whisper_encoder = model_whisper_encoder
        
        # 加入卷积层来缩小维度
        self.conv = nn.Conv1d(in_channels=1280, out_channels=1280, kernel_size=50, stride=50)
        
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
        stride = 50
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


if __name__ == "__main__":
    device = "cuda:0"
    path_llama = "Llama-2-7b-chat-hf"
    model_llama = AutoModelForCausalLM.from_pretrained(path_llama, torch_dtype=torch.float16).to(device)
    llama_embedding = model_llama.get_input_embeddings().to(device)
    tokenizer_llama = AutoTokenizer.from_pretrained(path_llama)
    
    path_whisper = "whisper-large-v3-turbo"
    model_whisper_encoder = AutoModelForSpeechSeq2Seq.from_pretrained(
        path_whisper, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
    ).model.encoder.to(device)
    processor_whisper = AutoProcessor.from_pretrained(path_whisper)

    lora_dir = "./saved_model/lora_weights"
    model_lora = PeftModel.from_pretrained(model_llama, lora_dir)

    # 加载multi_model可训练权重
    model_state_path = "./saved_model/trainable_multi_model.pth"
    trainable_state_dict = torch.load(model_state_path)

    # 初始化multi_model
    model_final = multi_model(model_lora, model_whisper_encoder, mode_train=False).to(torch.float16).to(device)
    model_final.load_state_dict(trainable_state_dict, strict=False)
    print("Model reloaded successfully.")
    
    with torch.no_grad():
        batch_size = 1
        model_final.eval()
        
        audio_file = "./dataset/audio/18.mp3"
        # 16.mp3
        array, _ = librosa.load(audio_file, sr=16000)
        input_features = processor_whisper(array, sampling_rate=16000, return_tensors="pt").input_features.to(torch.float16).to(device)
        
        user_prefix = "<s> [INST]"
        input_ids_user_prefix = tokenizer_llama(user_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        input_embedding_user_prefix = llama_embedding(input_ids_user_prefix).repeat(batch_size, 1, 1)
        
        assistant_prefix = "[/INST]"
        input_ids_assistant_prefix = tokenizer_llama(assistant_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][..., 1:].to(device)
        input_embedding_assistant_prefix = llama_embedding(input_ids_assistant_prefix).repeat(batch_size, 1, 1)
        
        input_embedding = None
        final_input = model_final(input_features, input_embedding, input_embedding_user_prefix, input_embedding_assistant_prefix, None, None, None)
        # print("问题开始", input_embedding_user_prefix)
        # print("回答开始", input_embedding_assistant_prefix)
        
        print(tokenizer_llama.batch_decode(model_lora.generate(inputs_embeds=final_input, do_sample=False, max_new_tokens=100)))

# 语音分析课程大作业2024秋
## 概述
该仓库前半部分为whisper做asr后输入LLM的结果。后半部分结合[openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)模型和[meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)模型，完成一个简单的语音文本多模态大模型的训练，由于训练数据和显卡的限制，后半部分仅作示范。
# Part1
## 环境配置
```bash
conda create -n multi python=3.10
conda activate multi
pip install -r requirements.txt
```
## 启动代码
```bash
cd web_app
python app.py
```
启动成功后在本地浏览器输入 http://127.0.0.1:5000/static/index.html 即可进入对话。

## 演示视频
<video src="./演示demo.mp4"></video>

# Part2
## 环境配置
```bash
conda create -n multi python=3.10
conda activate multi
pip install -r requirements.txt
```

## 数据集以及模型训练结果
### 数据集路径
- `/dataset`: 包含mp3文件和文本数据，mp3文件需要从[huggingface仓库](https://huggingface.co/datasets/weiyezhimeng/audio-text-dataset)获取。
我们的中文数据集从[这个仓库](https://github.com/hikariming/chat-dataset-baseline)获取。
### 模型训练结果
- 我们的训练结果在`./saved_model`中。

## 训练代码
```bash
python main.py --llama_path <你的Llama模型路径> --whisper_path <你的Whisper模型路径>
```
## 参数介绍
- `--llama_path`: 你下载llama到本地的路径
- `--whisper_path`: 你下载whisper到本地的路径
- `--device`: 单卡训练位置
- `--save_dir`: 模型保存路径
- `--train_file_path`: 训练数据集
余下部分为optimizer的训练参数和[lora](https://arxiv.org/abs/2106.09685)的训练参数。

## 推理代码
```bash
python inference_main.py --llama_path <你的Llama模型路径> --whisper_path <你的Whisper模型路径> --audio_file_path <你的mp3文件> --max_new_tokens <回答最大token数>
```
其余参数与main.py类似

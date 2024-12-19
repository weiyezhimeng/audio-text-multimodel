import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from evaluate import load
from evaluate import list_evaluation_modules
import json
from tqdm import tqdm
from mutagen.mp3 import MP3
def get_mp3_duration(file_path):
    audio = MP3(file_path)
    return audio.info.length

cer_metric = load("./cer")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

with open("./dataset/alpaca_data-0-3252-中文-已完成.json", 'r') as file:
    data = json.load(file)

normalizer = BasicTextNormalizer()
all_cer = 0
all_data_length = 0
for item in tqdm(data):
    if get_mp3_duration(item["path"]) > 30:
        continue
    all_data_length += 1
    audio_file_path = item["path"]
    result = pipe(audio_file_path)["text"]
    normalized_result = normalizer(result)
    normalized_label = normalizer(item["instruction"] + item["input"])
    cer = cer_metric.compute(references=[normalized_label], predictions=[normalized_result])
    print(normalized_result)
    print(normalized_label)
    print(cer)
    all_cer += cer 
print(all_cer/all_data_length)




from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM, AutoTokenizer
import os

device = "cuda:2"
path_llama = "/data2/hf_hub/models--meta-llama--Llama-2-7b-chat-hf"
model_llama = AutoModelForCausalLM.from_pretrained(path_llama, torch_dtype=torch.float16).to(device)
tokenizer_llama = AutoTokenizer.from_pretrained(path_llama)

model_id = "/data2/zhaojiawei_huggingface_hub/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# history
chat = []

def asr(file_path):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=device,
    )
    return pipe(file_path)["text"]

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    audio_file = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    # Speech recognition
    text_input = asr(file_path)
    user_chat = {"role": "user", "content": text_input}
    chat.append(user_chat)
    
    final_input = tokenizer_llama.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    final_prompt = tokenizer_llama.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    assistant_anser = tokenizer_llama.batch_decode(model_llama.generate(**final_input, do_sample=False, max_new_tokens=2048))[0][len(final_prompt)+len("]  "):-len("</s>")]
    assistant_chat = {"role": "assistant", "content": assistant_anser}
    chat.append(assistant_chat)
    print(chat)
    
    os.remove(file_path)  # Clean up after processing
    return jsonify({'text': assistant_anser})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
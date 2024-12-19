import argparse
from inference import inference

def get_args():
    parser = argparse.ArgumentParser(description="train manager")
    parser.add_argument("--llama_path", type=str, default="")
    parser.add_argument("--whisper_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./saved_model")
    parser.add_argument("--audio_file_path", type=str, default="./dataset/audio/1.mp3")
    
    # 模型生成结果参数
    parser.add_argument("--max_new_tokens", type=int, default=100)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inference(args)

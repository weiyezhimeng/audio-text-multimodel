import argparse
from train import train
def get_args():
    parser = argparse.ArgumentParser(description="train manager")
    parser.add_argument("--llama_path", type=str, default="")
    parser.add_argument("--whisper_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--train_file_path", type=str, default="./dataset/alpaca_data-0-3252-中文-已完成.json")

    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--Epoch", type=int, default=10)
    parser.add_argument("--accum_iter", type=int, default=32)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    train(args)

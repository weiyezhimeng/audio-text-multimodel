from torch.utils.data import Dataset, DataLoader

class mydataset(Dataset):
    def __init__(self, train_data):
        self.all = train_data
    def __len__(self):
        return len(self.all)
    def __getitem__(self, idx):
        # 加载 MP3 文件，指定采样率为 16000
        return self.all[idx]["input_features"], self.all[idx]["output"] + "</s>"

def load_data(train_data, batch):
    train_dataset = mydataset(train_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    return train_loader

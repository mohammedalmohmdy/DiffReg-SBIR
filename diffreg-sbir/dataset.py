import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def default_transform():
    return T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

class SketchPhotoDataset(Dataset):
    def __init__(self, root, pairs_file, transform=None):
        self.root = root
        self.transform = transform or default_transform()
        pairs_path = os.path.join(root, pairs_file)
        assert os.path.exists(pairs_path), f"{pairs_path} not found"
        with open(pairs_path, "r") as f:
            lines = f.read().splitlines()
        self.samples = []
        for L in lines:
            sk, ph, lbl = L.strip().split()
            self.samples.append((sk, ph, int(lbl)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        skp, php, lbl = self.samples[idx]
        sk = Image.open(os.path.join(self.root, skp)).convert("RGB")
        ph = Image.open(os.path.join(self.root, php)).convert("RGB")
        return self.transform(sk), self.transform(ph), lbl
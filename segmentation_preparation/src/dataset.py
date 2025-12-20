from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CorrosionDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.transform = transform

        with open(txt_file, "r") as f:
            lines = f.readlines()

        self.images = []
        self.masks = []

        for line in lines:
            img_path, mask_path = line.strip().split()
            self.images.append(img_path)
            self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        img = img.resize((512, 512))  # fixed resize
        mask = mask.resize((512, 512))  # fixed resize

        img = np.array(img)
        mask = np.array(mask)

        # Convert mask to [0,1]
        mask = (mask > 128).astype(np.int64)

        if self.transform:
            img = self.transform(Image.fromarray(img))

        return img, mask

import os
from PIL import Image
from torch.utils.data import Dataset


class ButterflyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )

        for label, class_name in enumerate(self.class_names):
            class_path = os.path.join(root_dir, class_name)
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.images.append(os.path.join(class_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

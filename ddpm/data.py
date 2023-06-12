from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import transforms

class CTLDRDataset(Dataset):
    def __init__(self, img_dir="/root/lmy/data/CTLDR/256_1mm_soft/full/L067"):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        assert len(self.img_paths) > 30,"数据集数量不足"
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5),(0.5))
        ])
        
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        image = self.transform(image)
        return image


if __name__ == "__main__":
    dataset = CTLDRDataset()
    data = dataset[0]
    print(len(dataset))
    pass
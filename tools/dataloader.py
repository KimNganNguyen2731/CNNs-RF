import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

def data_loader(batch_size: int = 64, resize: int = 224):
    """
    Data loading and data preprocessing.
    """
    
    DATA_DIR = "data/fruit_dataset"
    
    # Create transform for data
    transform = {
        "train": transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor()
        ]),
        "test": transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor()
        ]) 
    }
    
    # Train, test folder
    sets = ["train", "test"]
    
    # Create ImageFolder 
    image_data = {x: datasets.ImageFolder(os.path.join(DATA_DIR,x), 
                                          transform[x]) for x in sets}
    
    # DataLoader
    dataloaders = {
        x: DataLoader(image_data[x], batch_size=batch_size, shuffle=True, num_workers=4) 
        for x in sets
    }
    num_classes = image_data["train"].classes
    
    return dataloaders, num_classes

from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset,Dataset
import random
from PIL import Image
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label as it's not used

def get_data_loader(path, batch_size, num_samples=None, shuffle=True):
    # Define your transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.7002, 0.6099, 0.6036), (0.2195, 0.2234, 0.2097))  # Adjust these values if you have RGB images
    ])
    
    # Get the list of all image files in the root directory, excluding non-image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_extensions)]
    
    if len(image_files) == 0:
        raise ValueError("No valid image files found in the specified directory.")

    # If num_samples is not specified, use the entire dataset
    if num_samples is None or num_samples > len(image_files):
        num_samples = len(image_files)
    elif num_samples <= 0:
        raise ValueError("num_samples should be a positive integer.")

    print("data length: ", len(image_files))
    
    # Generate a list of indices to sample from (ensure dataset size is not exceeded)
    if shuffle:
        indices = random.sample(range(len(image_files)), num_samples)
    else:
        indices = list(range(num_samples))
    
    # Create the subset dataset
    subset_dataset = CustomDataset([image_files[i] for i in indices], transform=transform)
    
    # Create a DataLoader for the subset
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_img_tensors_as_grid(img_tensors, nrows, f, mean=[0.7002, 0.6099, 0.6036], std=[0.2195, 0.2234, 0.2097]):
    
    imgs_array = img_tensors.detach().cpu().numpy()

    # De-normalize the images
    mean = np.array(mean).reshape(1, 3, 1, 1)  # Reshape mean to (1, 3, 1, 1)
    std = np.array(std).reshape(1, 3, 1, 1)    # Reshape std to (1, 3, 1, 1)
    imgs_array = imgs_array * std + mean
    imgs_array = np.clip(imgs_array, 0, 1)
    imgs_array = (imgs_array * 255).astype(np.uint8)

    batch_size, _, img_size, _ = img_tensors.shape
    ncols = batch_size // nrows

    # Update img_arr to store the full grid
    img_arr = np.zeros((nrows * img_size, ncols * img_size, 3), dtype=np.uint8)

    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size

        # Ensure that we assign the correctly transposed image in (H, W, C) format
        img_arr[row_start:row_end, col_start:col_end, :] = np.transpose(imgs_array[idx], (1, 2, 0))

    # Save the final image grid
    Image.fromarray(img_arr, "RGB").save(f"{f}.jpg")

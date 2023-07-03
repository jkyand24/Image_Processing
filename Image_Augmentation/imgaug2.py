import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt

# transform_data 함수 정의하기

def imgaug_transform(image):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.Multiply((0.8, 1.2))
    ])
    
    image_np = image.permute(1, 2, 0).numpy() # tensor -> numpy
        # image_np.shape: (32, 32, 3)
    image_aug = seq(image=image_np)
    image_aug_copy = image_aug.copy()
    image_aug_tensor = torch.from_numpy(image_aug_copy).permute(2, 0, 1) # numpy -> tensor
        # image_aug_tensor.shape: [3, 32, 32]
    
    return image_aug_tensor

def transform_data(image):
    tensor = transforms.ToTensor()(image) # 인자를 tensor 형태로 전환
        # tensor.shape: [3, 32, 32]
    transformed_tensor = imgaug_transform(tensor) # tensor에 imgaug_transform 적용
    
    return transformed_tensor

# data set

train_dataset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform_data)

# data loader

batch_size = 4

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for images, labels in train_dataloader:
    fig, axes = plt.subplots(1, batch_size, figsize=(12, 4))
    
    for i in range(batch_size):
        image = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {labels[i]}")
        
    plt.show()
    
    break
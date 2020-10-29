import torch
import torchvision
import torchvision.transforms as transforms



imagenet_data = torchvision.datasets.ImageNet('../imagenet_data/', download=True)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers= 4)

data, labels = iter(data_loader).next()

print(data)
print(labels)


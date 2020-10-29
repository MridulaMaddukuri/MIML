import os
import itertools
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import tarfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

random.seed(seed)

print(os.getcwd())
# os.setcwd


## TODO: do transform for CIFAR : DONE
## TODO: Fix seed/ reproducability issue 
## Change format of the label : from roof shit

def get_train_val_split_data(root , transform =None, val_size = 0.33, seed = 1, num_workers = 4):

    trainset = datasets.CIFAR10(root, train=True,
                                        download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=trainset.__len__(),
                                  num_workers= num_workers)
      
    batch_data, batch_labels = iter(trainloader).next()
    
    # for (batch_data, batch_labels) in trainloader:
    train_data, val_data, train_labels, val_labels = train_test_split(batch_data,
                                                                          batch_labels,
                                                                          test_size = val_size,
                                                                          stratify=batch_labels,
                                                                          random_state =np.random.RandomState(seed))
    
    return (train_data, train_labels), (val_data, val_labels)




def get_test_data(root, transform =None, num_workers = 4):
  testset = datasets.CIFAR10(root, train=False, download=True, transform=transform)
  testloader = DataLoader(testset, batch_size = testset.__len__(), num_workers = num_workers)
  batch_data, batch_labels = iter(testloader).next()
  
  return batch_data, batch_labels 


def collate_fn(batch, input_size=32):
    img_lists, bag_labels = zip(*batch)

    imgs = [img for img_list in img_lists for img in img_list]

    sizes = torch.LongTensor([len(img_list) for img_list in img_lists])

    bag_labels = torch.stack(bag_labels)
    

    if len(imgs) != 0:
        imgs = torch.stack(imgs)
    else:
        imgs = torch.zeros((0, 3, input_size, input_size)).float()

    return imgs, sizes, bag_labels
    

class MIMLBagsData(Dataset):

    def __init__(self, 
                 data, # tuple
                 num_bag=250,
                 seed=1,
                 # TODO SW: consider making m { 'cata' : 10, 'catb' : 4, ...}
                 m=4, # Upper bound on number of instances per category per bag
                 category_list = None, # len(category_list) is N
                 transform=None,
                 num_workers=2):
        
        self.imgs, self.labels = data
        self.labels = [int(l) for l in self.labels]

        self.m = m
        self.num_bag = num_bag 
        # self.r = np.random.RandomState(seed)
        self.category_list = category_list
        if not self.category_list:
            self.category_list = list(set(self.labels))
        self.num_classes = len(self.category_list)
        self.num_workers = num_workers
        
        self.bags_list, self.labels_list = self._create_bags()

        

    def __len__(self):
        return len(self.labels_list)

    def _create_bags(self):
        label_img_dict = defaultdict(list)
        assert len(self.imgs) == len(self.labels), "There's a length mismatch"
        
        
        for i in range(len(self.imgs)):
            label_img_dict[self.labels[i]].append(self.imgs[i])


        
        num_categories = len(self.category_list)
        
        bags_list = []
        labels_list = []

        print('Creating bags .. \n')
        for i in tqdm(range(self.num_bag)):
            current_bag = []
            current_labels = []
            current_bag_labels = np.zeros(self.num_classes)
            # pick a random number indicating how many categories to sample instances from
            category_count = np.random.randint(low=1, high=num_categories//2)
            # print(f"category_count: {category_count}")
            sampled_categories = random.sample(self.category_list, category_count)

            for sc in sampled_categories:
              # pick number of instances per category per bag
              num_instances = np.random.randint(low=1, high=self.m)
              # print(f"Population: {len(label_img_dict[sc])} and Sample {num_instances}")
              current_bag += list(random.sample(label_img_dict[sc], num_instances))
              current_labels.append(sc)

            current_bag_labels[current_labels] = 1
            current_bag_labels = torch.from_numpy(current_bag_labels)
            bags_list.append(current_bag)
            labels_list.append(current_bag_labels)

        

        return bags_list, labels_list

    def __getitem__(self, index):
        return self.bags_list[index], self.labels_list[index]









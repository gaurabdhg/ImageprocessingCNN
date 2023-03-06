import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#Hyper-parameters 
epoch_count=25
batch_size=32


# for training:
learning_rate = 0.001
momentum = 0.9

torch.manual_seed(143)
val_size = 1000


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
tfs=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=tfs, download=True)

test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False,transform=tfs)

print(len(train_set))
print(len(test_set))

#verification :bonus task ::: to be completed without applying normalization transform.
def bonustask1():
    imgs = [item[0] for item in train_set] 
    imgs = torch.stack(imgs, dim=0).numpy()
    mu=[0,0,0]
    std=[0,0,0]
    for i in range(3):
      mu[i] = imgs[:,i,:,:].mean()
      std[i] = imgs[:,i,:,:].std()
    return mu,std

mu,sd=bonustask1()
print(f"Mean : {mu}   STD: {sd}")


#datasplit
train_size = len(train_set) - val_size
train_data, val_data = torch.utils.data.random_split(train_set, [train_size, val_size])

ids=torch.randperm(len(train_set))

split=1000
tr_ids,val_ids=ids[split:],ids[:split]

tr_sampler=SubsetRandomSampler(tr_ids)
val_sampler=SubsetRandomSampler(val_ids)


#Dataloaders
train_loader = DataLoader(train_set, batch_size, pin_memory=True,sampler=tr_sampler)
val_loader = DataLoader(train_set, batch_size*2,pin_memory=True,sampler=val_sampler)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,pin_memory=True)

#module for printing images in task 1.1
printsampler=DataLoader(train_set,5,sampler=tr_sampler)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def grid_plot(img):
    plt.figure(figsize=(20,14))
    plt.axis('off')
    plt.imshow(img.permute(1, 2, 0))
    #plt.savefig(f'grid.png')
    plt.show()
    
    
model=ImgClassifier()
model = model.to(device)  # put all model params on GPU.

# Create loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

print(model)

#variables for tracking the losses and accuracies
foo1=[]#train_loss
foo2=[]#train_accuracy
foo3=[]#valid_loss
foo4=[]#valid_accuracy
bestmodel={'epoch':  0,
        'model_state_dict': model.state_dict(),
        'loss':   0,
        'accuracy':  0
        }

train_evaluate()

import pdb
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support
torch.manual_seed(0)
import numpy as np; np.random.seed(0)

dataset = torchvision.datasets.ImageFolder(
    root='archive/TB_Chest_Radiography_Database',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
)
print(dataset.class_to_idx)
#validationset = torchvision.datasets.ImageFolder(
#    root='archive/TB_Chest_Radiography_Database',
#    transform=torchvision.transforms.ToTensor()
#)
y = 0
if y == 0:
    print('y is 0')
elif y == 1:
    print('y is 1')

val_size = int(0.2 * len(dataset))
train_size = int(len(dataset) - val_size)

train_data, val_data = random_split(dataset, [train_size, val_size])

# N, T, T, N, N
train_list_of_weights = []
train_data_images_order = [dataset.imgs[x][1] for x in train_data.indices]
for train_image in train_data_images_order:
    if train_image == 0:
        train_list_of_weights.append(1)
    elif train_image == 1:
        train_list_of_weights.append(5)

val_list_of_weights = []
val_data_images_order = [dataset.imgs[x][1] for x in val_data.indices]
for val_image in val_data_images_order:
    if val_image == 0:
        val_list_of_weights.append(1)
    elif val_image == 1:
        val_list_of_weights.append(5)

train_sample_weights = torch.utils.data.sampler.WeightedRandomSampler(torch.Tensor(train_list_of_weights), len(train_list_of_weights))
val_sample_weights = torch.utils.data.sampler.WeightedRandomSampler(torch.Tensor(val_list_of_weights), len(val_list_of_weights))
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=8, sampler=train_sample_weights)
validation_dataloader = torch.utils.data.DataLoader(val_data, batch_size=8, sampler=val_sample_weights)

import torch.nn as nn
import torch.nn.functional as F

net = models.resnet18()
net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
import torch.optim as optim
cost_function = nn.CrossEntropyLoss()#weight=torch.Tensor((1,5)))
gradient_descent = optim.Adam(net.parameters())#, lr=0.001)

writer = SummaryWriter('new_runs/control_resnet18')
count = 0

for epoch in range(20):  # loop over the dataset multiple times
    for i, training_data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        training_data_inputs, training_data_labels = training_data
        
        # zero the parameter gradients
        gradient_descent.zero_grad()

        # forward + backward + optimize
        model_predictions = net(training_data_inputs)
        cost = cost_function(model_predictions, training_data_labels)
        cost.backward()
        gradient_descent.step()
        count += 1
        writer.add_scalar('training loss', cost, count)
        print(cost)

        ## loading one batch of validation data
        validation_data = iter(validation_dataloader).next()

        ## ?? ... 
        validation_data_inputs, validation_data_labels = validation_data

        validation_model_predictions = net(validation_data_inputs)
        cost = cost_function(validation_model_predictions, validation_data_labels)
        writer.add_scalar('validation loss', cost, count)
        print(cost)

        #pdb.set_trace()
        precision, recall, fscore, support = precision_recall_fscore_support(training_data_labels, torch.argmax(model_predictions, dim=1), labels=(0,1), zero_division=0, average='binary')
        ## plot precision, recall, fscore - tensorboard
        writer.add_scalar('precision', precision, count)
        writer.add_scalar('recall', recall, count)
        writer.add_scalar('fscore', fscore, count)

        ## also plot validation precision, recall, fscore (which you haven't calculated yet)
        validation_precision, validation_recall, validation_fscore, validation_support = precision_recall_fscore_support(validation_data_labels, torch.argmax(validation_model_predictions, dim=1), labels=(0,1), zero_division=0, average='binary')
        writer.add_scalar('validation_precision', validation_precision, count)
        writer.add_scalar('validation_recall', validation_recall, count)
        writer.add_scalar('validation_fscore', validation_fscore, count)

        ## calculate the cost function for the validation data

writer.close()
print('Finished Training')
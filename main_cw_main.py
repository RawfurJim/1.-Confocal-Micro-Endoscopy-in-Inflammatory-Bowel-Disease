import scipy
import sklearn
from sklearn.feature_extraction import image
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import scikitplot as skplt

from mat4py import loadmat
import glob
import os
import cv2


data = pd.read_csv('data.csv')

Type_Class = np.where(data.Type == 'Benign' , 0, 1)


data['Type_Class'] = Type_Class


os.chdir(r'C:\Users\HP\Desktop\ml_cw\pol\MI_Data')
mat_files = glob.glob('*.mat')

Main_data = []
c=0


for fname in mat_files:
    
    
    if(data['fp'][c] == fname ):
        
        mat = scipy.io.loadmat('{0}'.format(fname) )
        
        
        g = np.array(mat['xstack'])
        
        images = transforms.ToTensor()
        i = images(g)
        #torch_tensor = torch.from_numpy(g).long()
        d=  i.unsqueeze(1)
        if(data['Type_Class'][c] == 1):
            l = np.array([0,1])
            Main_data.append([d,l])
        else:
            l = np.array([1,0])
            Main_data.append([d,l])
        c=c+1

batch_size = 1
learning_rate = 0.001
num_epochs = 50



           


train_dataset = Main_data[0:180]
test_dataset = Main_data[181:206]
#for i,l in a:
  #  print(l)
    


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(16*137*137, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)



    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))



        x = x.view(-1,16*137*137)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)




        return x



model = ConvNet()



#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


# training loop

# training loop
n_total_steps = len(train_dataset)
t = []
for epoch in range(num_epochs):
    for i,(images, l) in enumerate(train_dataset):
        images = images
       # np.array(l, dtype=np.float).shape = ()
        lebels = torch.from_numpy(l)
        h = lebels.float()
       # x = lebels.type(torch.LongTensor)
      
        outputs = model(images)
        

        p=nn.functional.softmax(outputs,dim=1)
      

        pm=p.mean(dim=0)
        pm = pm.float()
        
        #
        lo =  nn.MSELoss()
        loss = lo(pm, h)
        
        l = loss.detach().numpy()
        if(i+1) % 100 == 0:
            print(epoch,  l.item())
            t.append(l.item())
        
        
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

plt.figure(figsize=(10,5))


plt.plot(t, '-o', label="train")

plt.ylabel("Weighted x-entropy")
plt.title("Loss change over epoch")
plt.show()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    
   
    for i,(images, l) in enumerate(train_dataset):
        images = images
        lebels = torch.from_numpy(l)
        
        outputs = model(images)
        p=nn.functional.softmax(outputs,dim=1)
        pm=p.mean(dim=0)
        real_class = torch.argmax(lebels)
        

        predicted = torch.argmax(pm)
      
        

        if predicted == real_class:
            n_correct +=1
        n_samples +=1
        if i%20 == 0:
            acc = 100 * n_correct/ n_samples
            print('train',{i+1},acc)
    
        
        
 

    acc = 100 * n_correct/ n_samples
    print('train',acc)
            
        

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    
   
    for i,(images, l) in enumerate(test_dataset):
        images = images
        lebels = torch.from_numpy(l)
        
        outputs = model(images)
        p=nn.functional.softmax(outputs,dim=1)
        pm=p.mean(dim=0)
        real_class = torch.argmax(lebels)
        

        predicted = torch.argmax(pm)
        
        

        if predicted == real_class:
            n_correct +=1
        n_samples +=1
        if i%4 == 0:
            acc = 100 * n_correct/ n_samples
            print('train',{i+1},acc)
    
        
        
 

    acc = 100 * n_correct/ n_samples
    print('test',acc)
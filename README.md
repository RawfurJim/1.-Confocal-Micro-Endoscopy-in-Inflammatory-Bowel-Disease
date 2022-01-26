# 1.-Confocal-Micro-Endoscopy-in-Inflammatory-Bowel-Disease

# Dataset

In traditional image type dataset every image has a label but in this dataset the label was given by each patient and for each patient there can be 1-14 images. 


![image](https://user-images.githubusercontent.com/64610564/151218589-aded175e-947c-43f8-8066-3baf21e20ab0.png)
![image](https://user-images.githubusercontent.com/64610564/151218623-ba704102-9147-4cc0-bb3a-2b040ef72623.png)

I have image of two different patient. Here, 3 image join together for each patient. Throughout my dataset I have 1 to 12 image join together for each patient in a mat file.
To load the Dataset I have used following code


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

# Creating model
I have created CNN model with input channel 1 and two hidden layer. Here, Epoch = 50, Batch Size = 1, Learning Rate = 0.1


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(16*137*137, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        
 # Note
 Here, I did not provide my dataset. I have used a dataset provided by my University Professor. He did not give parmission to upload the dataset any parsonal website. 


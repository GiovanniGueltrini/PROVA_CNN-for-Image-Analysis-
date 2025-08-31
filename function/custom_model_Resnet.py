import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class CustomResNet(nn.Module):
    def __init__(self, n_classes,dropout_percentage):
        super(CustomResNet, self).__init__()
        
        self.dropout_percentage = dropout_percentage
        self.relu = nn.ReLU()
        
        # BLOCK-1 (starting block)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        # BLOCK-2 (1) 
        self.conv2_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_1_1 = nn.BatchNorm2d(64)  
        self.conv2_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        self.dropout2_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-2 (2)
        self.conv2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_2_1 = nn.BatchNorm2d(64)
        self.conv2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_2_2 = nn.BatchNorm2d(64)
        self.dropout2_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-3 (1) 
        self.conv3_1_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)
        self.concat_adjust_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout3_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-3 (2)
        self.conv3_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_2_1 = nn.BatchNorm2d(128)
        self.conv3_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_2_2 = nn.BatchNorm2d(128)
        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-4 (1) 
        self.conv4_1_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm4_1_1 = nn.BatchNorm2d(256)
        self.conv4_1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_1_2 = nn.BatchNorm2d(256)
        self.concat_adjust_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout4_1 = nn.Dropout(p=self.dropout_percentage)                                                 
        # BLOCK-4 (2)
        self.conv4_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_2_1 = nn.BatchNorm2d(256)
        self.conv4_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_2_2 = nn.BatchNorm2d(256)
        self.dropout4_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-5 (1)
        self.conv5_1_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm5_1_1 = nn.BatchNorm2d(512)
        self.conv5_1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_1_2 = nn.BatchNorm2d(512)
        self.concat_adjust_5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout5_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-5 (2)
        self.conv5_2_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_2_1 = nn.BatchNorm2d(512)
        self.conv5_2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_2_2 = nn.BatchNorm2d(512)
        self.dropout5_2 = nn.Dropout(p=self.dropout_percentage)

        # BLOCK-6 (1) 
        self.conv6_1_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm6_1_1 = nn.BatchNorm2d(1024)
        self.conv6_1_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm6_1_2 = nn.BatchNorm2d(1024)
        self.concat_adjust_6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout6_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-6 (2)
        self.conv6_2_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm6_2_1 = nn.BatchNorm2d(1024)
        self.conv6_2_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm6_2_2 = nn.BatchNorm2d(1024)
        self.dropout6_2 = nn.Dropout(p=self.dropout_percentage)

         # BLOCK-7 (1) 
        self.conv7_1_1 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm7_1_1 = nn.BatchNorm2d(2048)
        self.conv7_1_2 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm7_1_2 = nn.BatchNorm2d(2048)
        self.concat_adjust_7 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout7_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-7 (2)
        self.conv7_2_1 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm7_2_1 = nn.BatchNorm2d(2048)
        self.conv7_2_2 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm7_2_2 = nn.BatchNorm2d(2048)
        self.dropout7_2 = nn.Dropout(p=self.dropout_percentage)
        
        # Final Block input=(7x7) 
        self.avgpool = nn.AvgPool2d(kernel_size=(8,8), stride=(1,1))
        self.fc = nn.Linear(in_features=1*1*2048, out_features=1000, bias=True)
        self.out_1 = nn.Linear(in_features=1000, out_features=100,bias=True)
        self.out_2=nn.Linear(in_features=100, out_features=n_classes, bias=True) 
   
        
    def forward(self, x):
        
        # block 1 --> Starting block
        x = self.relu(self.batchnorm1(self.conv1(x))) #512x512 64
        op1 = self.maxpool1(x)  #512x512 64
        
        # block2 - 1
        x = self.relu(self.batchnorm2_1_1(self.conv2_1_1(op1)))    # conv2_1   
        x = self.batchnorm2_1_2(self.conv2_1_2(x))                 # conv2_1
        x = self.dropout2_1(x)
        # block2 - Adjust - No adjust in this layer as dimensions are already same
        # block2 - Concatenate 1
        op2_1 = self.relu(x + op1)
        
        # block2 - 2
        x = self.relu(self.batchnorm2_2_1(self.conv2_2_1(op2_1)))  # conv2_2 
        x = self.batchnorm2_2_2(self.conv2_2_2(x))                 # conv2_2
        x = self.dropout2_2(x)
        # op - block2
        op2 = self.relu(x + op2_1)
        

        # block3 - 1[Convolution block]
        x = self.relu(self.batchnorm3_1_1(self.conv3_1_1(op2)))    # conv3_1
        x = self.batchnorm3_1_2(self.conv3_1_2(x))                 # conv3_1
        x = self.dropout3_1(x)
        # block3 - Adjust
        op2 = self.concat_adjust_3(op2) # SKIP CONNECTION
        # block3 - Concatenate 1
        op3_1 = self.relu(x + op2)
        # block3 - 2[Identity Block]
        x = self.relu(self.batchnorm3_2_1(self.conv3_2_1(op3_1)))  # conv3_2
        x = self.batchnorm3_2_2(self.conv3_2_2(x))                 # conv3_2 
        x = self.dropout3_2(x)
        # op - block3
        op3 = self.relu(x + op3_1)
                

        # block4 - 1[Convolition block]
        x = self.relu(self.batchnorm4_1_1(self.conv4_1_1(op3)))    # conv4_1
        x = self.batchnorm4_1_2(self.conv4_1_2(x))                 # conv4_1
        x = self.dropout4_1(x)
        # block4 - Adjust
        op3 = self.concat_adjust_4(op3) # SKIP CONNECTION
        # block4 - Concatenate 1
        op4_1 = self.relu(x + op3)
        # block4 - 2[Identity Block]
        x = self.relu(self.batchnorm4_2_1(self.conv4_2_1(op4_1)))  # conv4_2
        x = self.batchnorm4_2_2(self.conv4_2_2(x))                 # conv4_2
        x = self.dropout4_2(x)
        # op - block4
        op4 = self.relu(x + op4_1)
       
        
        # block5 - 1[Convolution Block]
        x = self.relu(self.batchnorm5_1_1(self.conv5_1_1(op4)))    # conv5_1
        x = self.batchnorm5_1_2(self.conv5_1_2(x))                 # conv5_1
        x = self.dropout5_1(x)
        # block5 - Adjust
        op4 = self.concat_adjust_5(op4) # SKIP CONNECTION
        # block5 - Concatenate 1
        op5_1 = self.relu(x + op4)
        # block5 - 2[Identity Block]
        x = self.relu(self.batchnorm5_2_1(self.conv5_2_1(op5_1)))  # conv5_2
        x = self.batchnorm5_2_1(self.conv5_2_1(x))                 # conv5_2
        x = self.dropout5_2(x)
        # op - block5
        op5 = self.relu(x + op5_1)
       

        #Block 6
        x = self.relu(self.batchnorm6_1_1(self.conv6_1_1(op5)))    # conv5_1
        x = self.batchnorm6_1_2(self.conv6_1_2(x))                 # conv5_1
        x = self.dropout6_1(x)
        # block6 - Adjust
        op5 = self.concat_adjust_6(op5) # SKIP CONNECTION
        # block6 - Concatenate 1
        op6_1 = self.relu(x + op5)
        # block6 - 2[Identity Block]
        x = self.relu(self.batchnorm6_2_1(self.conv6_2_1(op6_1)))  # conv5_2
        x = self.batchnorm6_2_1(self.conv6_2_1(x))                 # conv5_2
        x = self.dropout6_2(x)
        # op - block6
        op6 = self.relu(x + op6_1)
       

        #Block 7
        x = self.relu(self.batchnorm7_1_1(self.conv7_1_1(op6)))    # conv5_1
        x = self.batchnorm7_1_2(self.conv7_1_2(x))                 # conv5_1
        x = self.dropout7_1(x)
        # block6 - Adjust
        op6 = self.concat_adjust_7(op6) # SKIP CONNECTION
        # block6 - Concatenate 1
        op7_1 = self.relu(x + op6)
        # block6 - 2[Identity Block]
        x = self.relu(self.batchnorm7_2_1(self.conv7_2_1(op7_1)))  # conv5_2
        x = self.batchnorm7_2_1(self.conv7_2_1(x))                 # conv5_2
        x = self.dropout7_2(x)
        # op - block6
        op7 = self.relu(x + op7_1)
        


        
       
        x = self.avgpool(op7)
       
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc(x))
        x = self.relu(self.out_1(x))
        x = self.out_2(x)

        return x

import torch
from torch import nn


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)



class Decoder(nn.Module):
    def __init__(self,z_size):
        super(Decoder,self).__init__()
        self.block1=nn.ConvTranspose2d(z_size, 1024,kernel_size=4, stride=1)
        # self.batch_size=batch_size
        self.block2=nn.Sequential(
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.ConvTranspose2d(1024,512,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1),
        nn.Tanh()
        )
    
    def forward(self,x):
        x=self.block1(x)
        return self.block2(x)






class Encoder(nn.Module):
    def __init__(self,in_channels):
        super(Encoder,self).__init__()
        # self.batch_size=batch_size
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=4,padding=2,stride=2), # 64*64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            
            nn.Conv2d(64,128,kernel_size=4,padding=2,stride=2), #32*32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128,256,kernel_size=4,padding=2,stride=2), #16*16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256,512,kernel_size=4,padding=2,stride=2), #8*8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512,1024,kernel_size=4,padding=2,stride=2), #4*4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool2d(1),
           

            nn.Conv2d(1024,100,kernel_size=1)
        )

    def forward(self,x):
        batch_size = x.size(0)
        # print(self.block(x).size)
        return self.block(x)
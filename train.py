from random import shuffle
from symbol import parameters
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import weights_init, Encoder, Decoder
import argparse
import numpy as np
import torch.optim as opt
from torch.autograd import Variable
from dataset import load_data, normalize
from torchvision.utils import save_image


parser=argparse.ArgumentParser("DC_GAN")
parser.add_argument("--img_path",default="/home/udaykanth/project/ADRL/VAE_notme/data/celeba/img_align_celeba/img_align_celeba")
parser.add_argument("--batchsize",default=128)
parser.add_argument("--epochs",default=30)


def main():
    args=parser.parse_args()
    device="cuda:2"
    img_size=64
    z_size=100
    in_channels=3
    k=2
    model_D=Decoder(z_size).to(device)
    model_E=Encoder(in_channels).to(device)
    model_E.apply(weights_init)
    model_D.apply(weights_init)
    loss=nn.MSELoss().to(device)
    # g_opt=opt.Adam(model_D.parameters(),lr=0.0002, betas=(0.5,0.999))
    # d_opt=opt.Adam(model_D.parameters(),lr=0.0002, betas=(0.5,0.999))
    Ae_opt=opt.Adam(list(model_D.parameters())+list(model_E.parameters()),lr=0.0002,betas=(0.5,0.999))
    data=load_data(args.img_path)
    real_data=DataLoader(dataset=data,batch_size=args.batchsize,shuffle=True)
    ae_loss=[]
    max_loss=np.inf
    # fixed_noise = Variable(torch.FloatTensor(np.random.randn(args.batchsize,z_size,1,1))).to(device)
    for epoch in range(args.epochs):
        real_images=tqdm(real_data)
        model_D.train()
        model_E.train()
        batch_id=0
        for imgs in real_images:
            imgs=imgs.to(device)
            batch_id+=1
            model_D.zero_grad()
            model_E.zero_grad()
            latent_var=model_E(imgs).to(device)
            reconstructed_images=model_D(latent_var)
            model_loss=loss(imgs,reconstructed_images)
            model_loss.backward()
            Ae_opt.step()
            if(batch_id%1000==0):
                save_image((model_D(latent_var).data[:25]), "outputs/%d_gen.png" % (epoch), nrow=5,normalize=True)
                save_image((imgs.data[:25]), "outputs/%d_real.png" % (epoch), nrow=5,normalize=True)
            real_images.set_description(desc='[%d/%d] Loss_D: %.4f' % (epoch,args.epochs,model_loss.item()))
            k=model_loss.item()
            if(k<max_loss):
                torch.save(model_D.state_dict(),"model.pth")
                max_loss=k
                # torch.save(model_D.state_dict(),"saving/disc/disc_{}.pth".format(epoch))

        
if __name__=="__main__":
    main()
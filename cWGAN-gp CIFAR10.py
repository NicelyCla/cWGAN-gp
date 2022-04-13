import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from tqdm.autonotebook import tqdm
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import numpy as np
from torch.autograd import Variable


num_epochs = 1000
betas = (0.5, 0.999)
LEARNING_RATE = 1e-4

batch_size = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
z_dim = 100        # latent Space
c_dim = 3          # Image Channel
label_dim = 10     # label 
image_size = 32
beta1 = 0.5
PATH = "./generate/"

CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

generator_out_linear = 100 #l rumore che andrà al generatore sarà 100+generator_out_linear, deve essere >=10

# MNIST dataset
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_set = dset.CIFAR10(root='./cifar10_data/', 
                       train=True,
                       transform=transform, 
                       download=True)
 
test_set = dset.CIFAR10(root='./cifar10_data/',
                      train=False, 
                      transform=transform, 
                      download=False)

train_loader = torch.utils.data.DataLoader(
    dataset = train_set,
    batch_size = batch_size,
    shuffle=True,
    drop_last=True
)
 
test_loader = torch.utils.data.DataLoader(
    dataset = test_set,
    batch_size = batch_size,
    shuffle=False,
    drop_last=True
)

# Generator model
class Generator(nn.Module):
    def __init__(self, z_dim, label_dim):
        super(Generator, self).__init__()

        self.ylabel=nn.Sequential(
            nn.Linear(10,generator_out_linear),
            nn.ReLU(True)
        )

        self.concat = nn.Sequential(
            #il rumore diventerà z_dim + il rumore della condizione
            nn.ConvTranspose2d(z_dim+generator_out_linear, 64*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),    
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),

            nn.ConvTranspose2d( 64*2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, y):
        
        y=y.reshape(-1,10)
        y = self.ylabel(y)
        y=y.reshape(-1,generator_out_linear,1,1)

        out = torch.cat([x, y] , dim=1)
        out=out.view(-1,100+generator_out_linear,1,1)

        out = self.concat(out)
        
        return out


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, nc=1, label_dim=10):
        super(Discriminator, self).__init__()
        
        self.ylabel=nn.Sequential(
            nn.Linear(10,32*32*1),
            nn.ReLU(True)
        )
        
        self.concate = nn.Sequential(
            #input size nc +  1 che è la condizione
            nn.Conv2d(nc+1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64*2 , 64*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64 * 4, 1, 4, 1, 0, bias=False),
        )
        
    def forward(self, x, y):
        
        y = y.reshape(batch_size,10)
        y = self.ylabel(y)
        y=y.view(-1,1,image_size,image_size)

        out = torch.cat([x, y] , dim=1)
        out = self.concate(out)

        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def gradient_penalty(critic, real, y_real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, y_real)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def train_GAN(dataset):
    for i,(real,label) in tqdm(enumerate(dataset)):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            critic.zero_grad()            
            ## Train with all-real batch        
            x_real = real.to(device)
            y_real = torch.ones(batch_size, ).to(device)
            label_onehot = onehot[label]
            critic_real = critic(x_real, label_onehot).reshape(-1)

            ## Train with all-fake batch
            noise = torch.randn(batch_size, z_dim, 1, 1, device = device)
            noise_label = label#(torch.rand(batch_size, 1) * label_dim).type(torch.LongTensor).squeeze()
            noise_label_onehot = onehot[noise_label].to(device)  # Genera label in modo casuale (-1,)
            x_fake = gen(noise, noise_label_onehot)       #Genera immagini false
            critic_fake = critic(x_fake, noise_label_onehot).reshape(-1)
            
            ## Train with gp
            gp = gradient_penalty(critic, x_real, label_onehot, x_fake, device=device)
            
            ##calculate loss
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            ##backward
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
             
        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen.zero_grad()
         
        noise = torch.randn(batch_size, z_dim, 1, 1, device = device)
        noise_label = (torch.rand(batch_size, 1) * label_dim).type(torch.LongTensor).squeeze()
        noise_label_onehot = onehot[noise_label].to(device)   # Genera label in modo casuale (-1,)
        
        x_fake = gen(noise, noise_label_onehot)
        gen_fake = critic(x_fake, noise_label_onehot).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        loss_gen.backward()
        opt_gen.step()

# Models
critic = Discriminator(c_dim, label_dim).to(device)
critic.apply(weights_init)

gen = Generator(z_dim, label_dim).to(device)
gen.apply(weights_init)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))


##creo un fixed_noise:
fixed_noise = torch.randn(100,100)
fixed_noise = fixed_noise.reshape(100,100,1,1)

##creo un fixed_label, in questo caso voglio una matrice 10*10 e un numero diverso da 1 a 9 nelle righe
labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).cuda() #labels = 00000000001111111111222222222233333333334444444444555555555566666666667777777777788888888889999999999
labels = labels.reshape(100,1)
one_hot = nn.functional.one_hot(labels, num_classes=10)#i labels codificato in one_hot
fixed_label = one_hot.reshape(100,10,1,1).float()

##creo il mio convertitore per i numeri da 0 a 9 in onehot:
onehot_before_cod = torch.LongTensor([i for i in range(10)]).cuda() #0123456789
onehot = nn.functional.one_hot(onehot_before_cod, num_classes=10)
onehot = onehot.reshape(10,10,1,1).float()

D_loss = []
G_loss = []

for epoch in tqdm(range(num_epochs)):
        
    # training
    train_GAN(train_loader)
    
    # test
    if epoch % 100 == 0 or epoch +1 == num_epochs:

        with torch.no_grad():
            out_imgs = gen(fixed_noise.to(device), fixed_label.to(device))

        save_image(out_imgs,f"{PATH}{epoch}.png", nrow = 10, normalize=True) #aggiungi percorso: "path/iterazione_classe.png" es "pippo/20000_3.png"

        #salva i modelli
        torch.save(critic.state_dict(),f'{PATH}discriminator_cDCGAN_{epoch}.pth')
        torch.save(gen.state_dict(), f'{PATH}generator_cDCGAN_{epoch}.pth')

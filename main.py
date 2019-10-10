from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from model import Generator, Discriminator

def make_datapath_list():

    train_img_list = list()
    for img_idx in range(200):
        img_path = './data/img_78/img_7_'+str(img_idx)+'.jpg'
        train_img_list.append(img_path)
        img_path = './data/img_78/img_8_'+str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list

class ImageTransform():

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    def __call__(self, img):
        return self.data_transform(img)


class GAN_img_Dataset(torch.utils.data.Dataset):

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_trans = self.transform(img)

        return img_trans


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(G, D, dataloader, num_epochs, device):


    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 =  0.0, 0.9

    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction = 'mean')


    z_dim = 20
    mini_batch_size = 64


    G = G.to(device)
    D = D.to(device)

    G.train()
    D.train()

    num_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []

    for epoch in range(num_epochs):

        epoch_g_loss = 0
        epoch_d_loss = 0

        for images in dataloader:
            if images.size()[0] == 1:
                continue

            imges = images.to(device)
            mini_batch_size = imges.size()[0]
            label_real = torch.full((mini_batch_size,),1).to(device)
            label_fake = torch.full((mini_batch_size,),0).to(device)

            d_out_real = D(imges)

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1,1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1,1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)


            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            g_loss.backward()
            g_optimizer.step()


            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        print(epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size)

    return G, D



def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        print('===>>> cuda')

    train_img_list = make_datapath_list()
    mean = (0.5,)
    std = (0.5,)

    train_dataset = GAN_img_Dataset(file_list=train_img_list,
            transform = ImageTransform(mean, std))

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle = True
            )

    G = Generator(z_dim =20, image_size=64)
    G.apply(weights_init)

    D = Discriminator(z_dim =20, image_size=64)
    D.apply(weights_init)


    num_epochs = 200
    print('===>>> start training')
    G_update, D_update = train(G, D, train_loader, num_epochs, device)

    print('===>>> finish training')


    batch_size = 8
    z_dim =20
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    fake_images = G_update(fixed_z.to(device))
    batch_iterator = iter(train_loader)
    imges = next(batch_iterator)

    fig = plt.figure(figsize=(15,6))
    for i in range(0,5):
        plt.subplot(2,5,i+1)
        plt.imshow(imges[i][0].cpu().detach().numpy(), 'gray')
        plt.subplot(2,5,5+i+1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')

        plt.savefig('output.jpg')

if __name__ == '__main__':
    main()

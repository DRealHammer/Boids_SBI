import torch
from torch import nn
from torch.nn import functional as F
from typing import *

import lightning as L


# from https://github.com/AntixK/PyTorch-VAE/blob/master/models
# define the LightningModule
class LightningVAE(L.LightningModule):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, lr: float = 1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

        
    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        param, img = batch
        mu, var = self.encode(img)
        x_hat = self.decode(mu)

        loss = self.loss_function(x_hat, img, mu, var, M_N=1)['loss']

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        param, img = batch
        mu, var = self.encode(img)
        x_hat = self.decode(mu)

        loss = self.loss_function(x_hat, img, mu, var, M_N=1)['loss']

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def latent(self, input: torch.Tensor) -> List[torch.Tensor]:
        result = self.encode(input)[0]
        
        return result
    


# define the LightningModule
class LightningSummaryFC(L.LightningModule):
    def __init__(self, in_dim: int, num_layers: int, hidden_dims: int, output_dims: int, lr: float = 1e-3, activation: str = 'relu', weight_decay: float = 0, loss_class=nn.MSELoss):
        super().__init__()

        self.save_hyperparameters()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.lr = lr
        self.weight_decay = weight_decay

        activation_fct = nn.ReLU

        if activation == 'tanh':
            activation_fct = nn.Tanh

        if activation == 'sigmoid':
            activation_fct = nn.Sigmoid

        # construct sequential
        layers = [nn.Flatten(), nn.Linear(in_dim, hidden_dims), activation_fct()]
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(activation_fct())

        layers.append(nn.Linear(hidden_dims, output_dims))

        self.network = nn.Sequential(*layers)

        self.loss = loss_class()

    def training_step(self, batch, batch_idx):

        param, img = batch

        if self.output_dims == 4:
            param = param[:, [2, 3, 4, 5]]

        param_hat = self.network(img)

        loss = self.loss(param, param_hat)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch):

        param, img = batch

        if self.output_dims == 4:
            param = param[:, [2, 3, 4, 5]]

        param_hat = self.network(img)

        loss = self.loss(param, param_hat)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def latent(self, input: torch.Tensor) -> List[torch.Tensor]:
        
        for module in self.network[:-1]:
            input = module(input)
        
        return input
    

class LightningSummaryConv(L.LightningModule):
    def __init__(self, in_channels: int, output_dims: int, hidden_dims: List = None, lr: float = 1e-3, activation: str = 'relu', weight_decay: float = 0, loss_class=nn.MSELoss):
        super().__init__()

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(nn.Flatten())
        modules.append(nn.Linear(hidden_dims[-1]*4, hidden_dims[-1]*4))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(hidden_dims[-1]*4, output_dims))
        self.network = nn.Sequential(*modules)

        self.save_hyperparameters()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss = loss_class()

    def training_step(self, batch, batch_idx):

        param, img = batch
        if self.output_dims == 4:
            param = param[:, [2, 3, 4, 5]]

        param_hat = self.network(img)

        loss = self.loss(param, param_hat)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch):

        param, img = batch
        if self.output_dims == 4:
            param = param[:, [2, 3, 4, 5]]

        param_hat = self.network(img)

        loss = self.loss(param, param_hat)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def latent(self, input: torch.Tensor) -> List[torch.Tensor]:
        
        for module in self.network[:-2]:
            input = module(input)
        
        return input
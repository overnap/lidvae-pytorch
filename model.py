import torch
from module import *


class VAE(torch.nn.Moduel):
    def __init__(self):
        super(VAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

    def loss(self, *args):
        raise NotImplementedError


class VanillaVAE(VAE):

    def __init__(
        self,
        in_channel=3,
        latent_channel=32,
        hidden_channels=[32, 64, 128],
        input_dim=28,
        beta=1.0,
        is_log_mse=False,
        dataset=None,
    ):
        """
        Conventional VAE with residual-conv encoder and MLP decoder, for image dataset.
        Note that decoder is 4-layer MLP, to avoid using convolution and its transpose.
        Beta and logMSE features are ready-to-use, but are disabled by default.
        """
        if dataset == "celeba":
            in_channel = 3
            latent_channel = 64
            hidden_channels = [32, 64, 128, 256]
            input_dim = 64
        elif dataset == "mnist":
            in_channel = 1
            latent_channel = 32
            hidden_channels = [32, 64, 128]
            input_dim = 28

        super(VAE, self).__init__()

        self.latent_channel = latent_channel
        self.beta = beta
        self.is_log_mse = is_log_mse

        fc_dim = input_dim
        transpose_padding = []
        for _ in range(len(hidden_channels)):
            transpose_padding.append((fc_dim + 1) % 2)
            fc_dim = (fc_dim - 1) // 2 + 1
        transpose_padding.reverse()

        # make encoder
        self.encoder = []
        last_channel = in_channel

        for channel in hidden_channels:
            self.encoder.append(
                torch.nn.Sequential(
                    ResidualBlock(last_channel, channel, 2),
                    ResidualBlock(channel, channel, 1),
                )
            )
            last_channel = channel

        self.encoder.append(
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(last_channel * (fc_dim**2), latent_channel * 2),
                torch.nn.BatchNorm1d(latent_channel * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_channel * 2, latent_channel * 2),
            )
        )
        self.encoder = torch.nn.Sequential(*self.encoder)

        # Make decoder
        self.decoder = []

        # First layer: half of final dimension
        last_channel = latent_channel
        channel = (input_dim**2) * in_channel // 2
        self.decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(last_channel, channel),
                torch.nn.BatchNorm1d(channel),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(channel, channel),
                torch.nn.BatchNorm1d(channel),
                torch.nn.LeakyReLU(),
            )
        )

        # Second and last layer: full dimension
        last_channel = channel
        channel = (input_dim**2) * in_channel
        self.decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(last_channel, channel),
                torch.nn.BatchNorm1d(channel),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(channel, channel),
                torch.nn.BatchNorm1d(channel),
                torch.nn.LeakyReLU(),
            )
        )

        # Unflatten to shape of image
        self.decoder.append(torch.nn.Unflatten(1, (in_channel, input_dim, input_dim)))

    def encode(self, input):
        ret = self.encoder(input)
        return ret.split(ret.shape[1] // 2, 1)

    def decode(self, input):
        return self.decoder(input)

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        return self.decode(z), mu, log_var

    def loss(self, input, output, mu, log_var):
        loss_recon = (
            ((input - output) ** 2).mean(dim=0).sum()
            if not self.is_log_mse
            else (
                0.5
                * torch.ones_like(input[0]).sum()
                * (
                    (
                        2 * torch.pi * ((input - output) ** 2).mean(1).mean(1).mean(1)
                        + 1e-5  # To avoid log(0)
                    ).log()
                    + 1
                )
            ).mean()
        )
        loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()

        return loss_recon + loss_reg * self.beta, loss_recon.detach(), loss_reg.detach()

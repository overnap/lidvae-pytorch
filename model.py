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

        # Make encoder
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
            )
        )

        # Unflatten to shape of image
        self.decoder.append(torch.nn.Unflatten(1, (in_channel, input_dim, input_dim)))

        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        self.decoder = torch.nn.Sequential(*self.decoder)

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


class LIDVAE(VAE):

    def __init__(
        self,
        in_channel=3,
        latent_channel=32,
        hidden_channels=[32, 64, 128],
        icnn_channels=[512, 512],
        input_dim=28,
        inverse_lipschitz=0.0,
        beta=1.0,
        is_log_mse=False,
        dataset=None,
    ):
        """
        LIDVAE with residual-conv encoder and Brenier map decoder, for image dataset.
        Decoder consists of 2 ICNN, so 2-length array is expected for hidden channels of ICNNs.
        See Wang et al. for details on Brenier map.
        Inverse Lipschitz, Beta, and logMSE features are ready-to-use, but are disabled by default.
        """
        if len(icnn_channels) != 2:
            raise ValueError("2-length array was expected for `icnn_channels`")

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
        self.il_factor = inverse_lipschitz / 2
        self.beta = beta
        self.is_log_mse = is_log_mse

        fc_dim = input_dim
        transpose_padding = []
        for _ in range(len(hidden_channels)):
            transpose_padding.append((fc_dim + 1) % 2)
            fc_dim = (fc_dim - 1) // 2 + 1
        transpose_padding.reverse()

        # Make encoder
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

        # First layer: ICNN in latent channel
        self.decoder.append(ICNN(latent_channel, icnn_channels[0]))

        # In the original implmentation,
        # a trainable full-rank matrix is used as Beta via SVD (as in appendix)
        # Here, we use an identity matrix for injective map (as in main text)
        self.decoder.append(
            torch.eye((input_dim**2) * in_channel, latent_channel, requires_grad=False)
        )

        # Second and last layer: ICNN in data dimension
        self.decoder.append(ICNN((input_dim**2) * in_channel, icnn_channels[1]))

        # Unflatten to shape of image
        self.decoder.append(torch.nn.Unflatten(1, (in_channel, input_dim, input_dim)))

        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        self.decoder = torch.nn.ModuleList(*self.decoder)

    def encode(self, input):
        ret = self.encoder(input)
        return ret.split(ret.shape[1] // 2, 1)

    def decode(self, input):
        # x is result of first ICNN
        x = self.decoder[0](input) + self.il_factor * input.pow(2).sum(1, keepdim=True)
        # x is result of brenier map
        x = torch.autograd.grad(x, [input], torch.ones_like(x))[0]
        # x is result of Beta (id mat)
        x = self.decoder[1](x)
        # y is result of second ICNN
        y = self.decoder[2](x) + self.il_factor * x.pow(2).sum(1, keepdim=True)
        # y is result of brenier map
        y = torch.autograd.grad(y, [x], torch.ones_like(y))[0]

        return self.decoder[3](y)

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


class ConvVAE(VAE):

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
        Conventional VAE with residual-convolution encoder and decoder, for image dataset.
        Note that decoder also consists of convolution and its transpose.
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

        # Make encoder
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
        hidden_channels.reverse()

        self.decoder = []
        last_channel = hidden_channels[0]

        self.decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(latent_channel, last_channel * (fc_dim**2)),
                torch.nn.BatchNorm1d(last_channel * (fc_dim**2)),
                torch.nn.LeakyReLU(),
                torch.nn.Unflatten(1, (last_channel, fc_dim, fc_dim)),
                ResidualBlock(last_channel, last_channel, 1),
            )
        )

        for channel, pad in zip(hidden_channels[1:], transpose_padding[:-1]):
            self.decoder.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(last_channel, channel, 3, 2, 1, pad),
                    torch.nn.BatchNorm2d(channel),
                    torch.nn.LeakyReLU(),
                )
            )
            last_channel = channel

        self.decoder.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    last_channel, last_channel, 3, 2, 1, transpose_padding[-1]
                ),
                torch.nn.BatchNorm2d(last_channel),
                torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d(last_channel, in_channel, 3, 1, 1),
            )
        )
        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        self.decoder = torch.nn.Sequential(*self.decoder)

        hidden_channels.reverse()

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

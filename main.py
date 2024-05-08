from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import torchvision
import os
import model as Model


def train_and_test(model: Model.VAE, epochs=50, batch_size=512, device="cuda"):
    transforms = torchvision.transforms.Compose(
        [
            # For MNIST
            torchvision.transforms.RandomRotation(20),
            torchvision.transforms.RandomResizedCrop((28, 28), (0.9, 1), (0.9, 1.1)),
            # For CelebA
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.CenterCrop(148),
            # torchvision.transforms.Resize(64),
            # For all
            torchvision.transforms.ToTensor(),
        ]
    )

    loader_train = DataLoader(
        torchvision.datasets.MNIST(root="C:/dataset/", transform=transforms),
        # torchvision.datasets.CelebA(root="./dataset", transform=transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    loader_test = DataLoader(
        torchvision.datasets.MNIST(
            root="C:/dataset/", transform=transforms, train=False
        ),
        # torchvision.datasets.CelebA(
        #     root="C:/dataset/", transform=transforms, split="test"
        # ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs * len(loader_train)
    )

    name = type(model).__name__ + datetime.now().strftime(" %m%d %H%M")
    name += " beta=" + str(float(model.beta))
    name += " log=" + str(model.is_log_mse)
    if type(model).__name__ == "LIDVAE":
        name += " il=" + str(float(model.il_factor))

    writer = SummaryWriter(log_dir="runs/" + name)
    if not os.path.exists("./result/"):
        os.mkdir("./result/")
    if not os.path.exists("./result/" + name):
        os.mkdir("./result/" + name)

    # Main loop
    for epoch in tqdm(range(epochs), desc=name):
        model.train()
        loss_total = 0.0
        loss_recon_total = 0.0
        loss_reg_total = 0.0

        # Train loop
        for x, y in tqdm(loader_train, leave=False, desc="Train"):
            x = x.to(device)
            y = y.to(device)

            result = model(x)
            loss, loss_recon, loss_reg = model.loss(x, *result)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_total += float(loss)
            loss_recon_total += float(loss_recon)
            loss_reg_total += float(loss_reg)

        writer.add_scalar("loss/train", loss_total / len(loader_train), epoch)
        writer.add_scalar("recon/train", loss_recon_total / len(loader_train), epoch)
        writer.add_scalar("reg/train", loss_reg_total / len(loader_train), epoch)

        model.eval()
        loss_total = 0

        # We cannot use no_grad, since LIDVAE requires calculation of gradient
        # with torch.no_grad():
        if True:
            # Validation loop
            for x, y in tqdm(loader_test, leave=False, desc="Evaluate"):
                x = x.to(device)
                y = y.to(device)

                result = model(x)
                loss = model.loss(x, *result)[0]
                loss_total += float(loss)

            # Save reconstruction example
            for _ in tqdm(range(1), leave=False, desc="Test"):
                x, _ = next(iter(loader_test))
                x = x.to(device)
                x.requires_grad = True

                result = model(x)
                save_image(
                    x[:256],
                    "./result/" + name + "/" + str(epoch) + "_origin.png",
                    normalize=True,
                    nrow=16,
                )
                save_image(
                    result[0][:256].clip(0, 1),
                    "./result/" + name + "/" + str(epoch) + "_recon.png",
                    normalize=True,
                    nrow=16,
                )

                # Save sampled example
                x = torch.randn((x.shape[0], model.latent_channel)).to(device)
                x.requires_grad = True
                result = model.decode(x)
                save_image(
                    result[:256].clip(0, 1),
                    "./result/" + name + "/" + str(epoch) + "_sample.png",
                    normalize=True,
                    nrow=16,
                )

        writer.add_scalar("loss/test", loss_total / len(loader_test), epoch)
        if epoch % 10 == 9:
            torch.save(
                model.state_dict(), "./result/" + name + "/model_" + str(epoch) + ".pt"
            )

    # Generate samples to calculate FID score
    # We cannot use no_grad, since LIDVAE requires calculation of gradient
    # with torch.no_grad():
    if True:
        if not os.path.exists("./result/" + name + "/generation"):
            os.mkdir("./result/" + name + "/generation")

        SAMPLE_ITERATION = 50
        for i in tqdm(range(SAMPLE_ITERATION), leave=False, desc="Generate"):
            x = torch.randn((batch_size, model.latent_channel)).to(device)
            x.requires_grad = True
            x = model.decode(x).clip(0, 1)

            for j in range(batch_size):
                save_image(
                    x[j],
                    "./result/"
                    + name
                    + "/generation/"
                    + str(i * batch_size + j)
                    + ".png",
                    normalize=True,
                    nrow=1,
                )

    writer.close()

    try:
        import pytorch_fid

        fid = os.popen(
            f'python -m pytorch_fid ./mnist/ "./result/{name}/generation/" --device cuda:0'
        ).read()
        print(fid)

    except ModuleNotFoundError:
        print("Please install `pytorch_fid` to show FID score")


if __name__ == "__main__":
    train_and_test(Model.LIDVAE(is_log_mse=True, inverse_lipschitz=0.0))

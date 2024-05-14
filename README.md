# lidvae-pytorch

Unofficial LIDVAE[1] implementation.  
Rewritten in PyTorch for basic image generation tasks.  
For the official TensorFlow implementation, see [here](https://github.com/yixinwang/lidvae-public).

The following features are also implemented:
- $\beta$ from $\beta$-VAE[2]
- $\log{\text{MSE}}$ which means optimal decoder variance from $\sigma$-VAE[3]
- Inverse lipschitz constraints from IL-LIDVAE[4]

### References

[1] Wang, Yixin, David Blei, and John P. Cunningham. "Posterior collapse and latent variable non-identifiability." Advances in Neural Information Processing Systems 34 (2021): 5443-5455.

[2] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with a constrained variational framework." ICLR (Poster) 3 (2017).

[3] Rybkin, Oleh, Kostas Daniilidis, and Sergey Levine. "Simple and effective vae training with calibrated decoders." International conference on machine learning. PMLR, 2021.

[4] Kinoshita, Yuri, et al. "Controlling posterior collapse by an inverse Lipschitz constraint on the decoder network." International Conference on Machine Learning. PMLR, 2023.

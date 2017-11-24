# Deep Generative Models

This repository is provides a standardized implementation framework for various popular decoder-based deep generative models. The following models are currently implemented

1. VAE
2. VAE (autoregressive inference)

In all cases, posterior regularization is applied to disentangle style (z) from content/label (y) on the SVHN dataset.

TODO:

1. AC-GAN/InfoGAN ([see this repo](https://github.com/RuiShu/acgan-biased))
2. BEGAN ([see this repo](https://github.com/RuiShu/began))
3. WGAN

## Dependencies

You'll need

```
tensorflow==1.1.0
scipy==0.19.0
tensorbayes==0.3.0
tensorflow==1.4.0
```

## Run models

All execution scripts adhere to the following format
```
python run_*.py --cmd
```

A list of possible commandline arguments can be found in each `run_*.py` script. In the case of VAE, pay attention to the choice of encoder/decoder architecture controlled by the argument `--design`, as this influences whether autoregression is applied during inference. Tensorboard logs are automatically saved to `./log/` and models are saved to `./checkpoints/`.

## Results

z-space and y-space interpolations provided respectively.

### VAE

<p align = 'center'>
<img src="assets/interp_by_z.gif" width="300">
<img src="assets/interp_by_y.gif" width="300">
</p>

### VAE (autoregressive inference)

No noticeable difference from vanilla VAE. I'd be curious to see if top-down inference makes a difference.

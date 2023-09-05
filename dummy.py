from multiprocessing import freeze_support

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


def main():
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=256,
        timesteps=1000,  # number of steps
        sampling_timesteps=250
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        r'D:\Creadto\Heritage\Dataset\skin-tester',
        train_batch_size=16,
        train_lr=8e-5,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=True  # whether to calculate fid during training
    )

    trainer.train()


if __name__ == '__main__':
    freeze_support()
    main()

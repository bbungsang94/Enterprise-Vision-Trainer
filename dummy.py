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
        timesteps=100,  # number of steps
        sampling_timesteps=25
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        r'D:\Creadto\Heritage\Dataset\skin-tester',
        train_batch_size=8,
        train_lr=8e-5,
        train_num_steps=2500,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=False,  # whether to calculate fid during training
        save_best_and_latest_only=False,
        save_and_sample_every=124
    )

    trainer.train()


if __name__ == '__main__':
    freeze_support()
    main()

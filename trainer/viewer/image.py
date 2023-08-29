import os

import imageio
import torch
import einops
import numpy as np
from matplotlib import pyplot as plt

from trainer.viewer.base import Base


class ImageViewer(Base):
    def show(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass

    def summary(self, **kwargs):
        pass


class DiffusionViewer(Base):
    def show(self, **kwargs):
        """Shows the provided images as sub-pictures in a square"""
        images = kwargs['images']
        title = "Show samples" if 'title' not in kwargs else kwargs['title']

        # Converting images to CPU numpy arrays
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()

        # Defining number of rows and columns
        fig = plt.figure(figsize=(8, 8))
        rows = int(len(images) ** (1 / 2))
        cols = round(len(images) / rows)

        # Populating figure with sub-plots
        idx = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, idx + 1)

                if idx < len(images):
                    plt.imshow(images[idx][0], cmap="gray")
                    idx += 1
        fig.suptitle(title, fontsize=30)

        # Showing the figure
        plt.show()

    def save(self, model, images,
             betas, alphas, alpha_bars,
             frames_per_gif, gif_name,
             save_path, **kwargs):
        n_samples = len(images)
        device = torch.device("cpu")
        c, w, h = images[0].shape

        """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
        frame_idxs = np.linspace(0, model, frames_per_gif).astype(np.uint)
        frames = []

        with torch.no_grad():
            # Starting from random noise
            x = torch.randn(n_samples, c, h, w).to(device)

            for idx, t in enumerate(list(range(model.n_steps))[::-1]):
                # Estimating noise to be removed
                time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
                eta_theta = model(x, time_tensor)

                alpha_t = alphas[t]
                alpha_t_bar = alpha_bars[t]

                # Partially denoising the image
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    z = torch.randn(n_samples, c, h, w).to(device)

                    # Option 1: sigma_t squared = beta_t
                    beta_t = betas[t]
                    sigma_t = beta_t.sqrt()

                    # Option 2: sigma_t squared = beta_tilda_t
                    # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                    # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                    # sigma_t = beta_tilda_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z

                # Adding frames to the GIF
                if idx in frame_idxs or t == 0:
                    # Putting digits in range [0, 255]
                    normalized = x.clone()
                    for i in range(len(normalized)):
                        normalized[i] -= torch.min(normalized[i])
                        normalized[i] *= 255 / torch.max(normalized[i])

                    # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                    frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                    frame = frame.cpu().numpy().astype(np.uint8)

                    # Rendering frame
                    frames.append(frame[:, :, 0])

        # Storing the gif
        with imageio.get_writer(os.path.join(save_path, gif_name), mode="I") as writer:
            for idx, frame in enumerate(frames):
                writer.append_data(frame)
                if idx == len(frames) - 1:
                    for _ in range(frames_per_gif // 3):
                        writer.append_data(frames[-1])
        return x

    def summary(self, **kwargs):
        self.save(**kwargs)

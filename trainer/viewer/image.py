import math
import os
from typing import Dict
import cv2
import imageio
import torch
import einops
import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torchvision.io import write_jpeg
from trainer.viewer.base import Base


class ImageViewer(Base):
    def show(self, **kwargs):
        pass

    def save(self, images: Dict[str, torch.Tensor], save_path: str):
        for filename, batch in images.items():
            grid = make_grid(batch, nrow=int(math.sqrt(len(batch)))).cpu() * 255
            write_jpeg(grid.type(dtype=torch.uint8), os.path.join(save_path, filename + ".jpg"))

    def summary(self, **kwargs):
        pass


class LandmarkViewer(Base):
    def show(self, **kwargs):
        pass

    def save(self, images: Dict[str, torch.Tensor], save_path: str):
        grid = make_grid(images['inputs'], nrow=int(math.sqrt(len(images['inputs'])))).cpu() * 255
        write_jpeg(grid.type(dtype=torch.uint8), os.path.join(save_path, "input_images" + ".jpg"))

        grid = make_grid(images['latent'], nrow=int(math.sqrt(len(images['latent'])))).cpu() * 255
        write_jpeg(grid.type(dtype=torch.uint8), os.path.join(save_path, "latent_images" + ".jpg"))

        for i in range(len(images['inputs'])):
            draw_image = self.draw(images['inputs'][i], images['outputs'][i], images['labels'][i], gap=True)
            cv2.imwrite(os.path.join(save_path, "sample-" + str(i) + ".jpg"), draw_image)

    def draw(self, image: torch.Tensor, a, b, gap=False):
        rows, cols = 1024, 1024
        image = image.cpu().numpy() * 255
        image = image.transpose(1, 2, 0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (1024, 1024))
        for a_point, b_point in zip(a, b):
            a_x = min(math.floor(a_point[0] * cols), cols - 1)
            a_y = min(math.floor(a_point[1] * rows), rows - 1)
            if a_point.shape[0] == 3:
                a_z = a_point[2]
            else:
                a_z = None
            b_x = min(math.floor(b_point[0] * cols), cols - 1)
            b_y = min(math.floor(b_point[1] * rows), rows - 1)
            if b_point.shape[0] == 3:
                b_z = b_point[2]
            else:
                b_z = None

            if a_z is None or b_z is None:
                depth = 255
            else:
                depth = int(torch.clamp(abs(a_z-b_z) * 5, min=-0.0, max=1.0) * 255)

            if gap:
                image = cv2.line(image, (a_x, a_y), (b_x, b_y), (0, 255, 0), 2)

            image = cv2.circle(image, (a_x, a_y), 4, (0, 0, depth), 4)
            image = cv2.circle(image, (b_x, b_y), 4, (depth, 0, 0), 4)

        return image

    def summary(self, **kwargs):
        pass

class DiffusionViewer(Base):
    def show(self, **kwargs):
        """Shows the provided images as sub-pictures in a square"""
        images = kwargs['images']
        gap = len(images)
        title = "Show samples" if 'title' not in kwargs else kwargs['title']

        # Converting images to CPU numpy arrays
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()

        # Defining number of rows and columns
        fig = plt.figure(figsize=(gap // 2, gap // 2))
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

    def save(self, model, x0, inputs,
             frames_per_gif, gif_name,
             save_path, **kwargs):

        n_samples, c, w, h = x0.shape

        """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
        frame_idxs = np.linspace(0, model.n_steps, frames_per_gif).astype(np.uint)
        frames = []

        with torch.no_grad():
            # Starting from random noise
            _, epsilon = model.make_noise(x0)

            for idx, t in enumerate(list(range(model.n_steps))[::-1]):
                # Estimating noise to be removed
                recon = model.regenerate(epsilon, t)

                # Adding frames to the GIF
                if idx in frame_idxs or t == 0:
                    # Putting digits in range [0, 255]
                    normalized = recon.clone()
                    for i in range(len(normalized)):
                        normalized[i] -= torch.min(normalized[i])
                        normalized[i] *= 255 / torch.max(normalized[i])

                    # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                    frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                    frame = frame.cpu().numpy().astype(np.uint8)

                    # Rendering frame
                    frames.append(frame)

        # Storing the gif and images
        with imageio.get_writer(os.path.join(save_path, gif_name), mode="I") as writer:
            for idx, frame in enumerate(frames):
                writer.append_data(frame)
                if idx == len(frames) - 1:
                    for _ in range(frames_per_gif // 3):
                        writer.append_data(frames[-1])

        x0_grid = make_grid(x0, nrow=int(math.sqrt(n_samples))).cpu() * 255
        x0_grid = x0_grid.type(dtype=torch.uint8)
        write_jpeg(x0_grid, os.path.join(save_path, "Last x0 images.jpeg"))
        input_grid = make_grid(inputs, nrow=int(math.sqrt(n_samples))).cpu() * 255
        input_grid = input_grid.type(dtype=torch.uint8)
        write_jpeg(input_grid, os.path.join(save_path, "Last input images.jpeg"))

    def summary(self, **kwargs):
        self.save(**kwargs)

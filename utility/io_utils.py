import os
import shutil


class FFHQHandler:
    @staticmethod
    def combine_images(root):
        folders = os.listdir(root)
        folders = [x for x in folders if os.path.isdir(os.path.join(root,x))]
        for folder in folders:
            sub_path = os.path.join(root, folder)
            images = os.listdir(sub_path)
            for image in images:
                shutil.move(os.path.join(sub_path, image), os.path.join(root, image))
            os.rmdir(sub_path)


if __name__ is "__main__":
    FFHQHandler.combine_images(r'D:\Creadto\Heritage\Dataset\ffhq-dataset\thumbnails128x128')

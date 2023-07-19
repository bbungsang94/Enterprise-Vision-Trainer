import pickle
import os
import pandas as pd
import numpy as np
import torch
from models.gender import MiVOLOClassifier
from facial_landmarks.cv_mesh.model import FaceLandMarks
from utility.monitoring import summary_device
from tqdm import tqdm
from torchvision.utils import save_image
from facial_landmarks.utility import make_3d_points
import cv2
import time


# 이미지 생성하고
# 성별 분류와 함께 랜드마크를 찍는다.
# neutral 0, male 1, female 2, age
# 20개의 shape / 10개의 expression / jaw rot 4 / neck 3
# FLAME의 랜드마크와 비교함
# Loss 계산
# 입력은 이미지 로스는 랜드마크 출력은 3d

class Actor:
    def __init__(self, generate_path, gender_args, params, **kwargs):
        self.device = summary_device()
        self._generator = self._load_generator(generate_path)
        self._gender = MiVOLOClassifier(**gender_args)
        self._landmarker = FaceLandMarks()
        self._params = params
        self._config = kwargs

        self._z_dim = self._generator.z_dim

    def generate_images(self):
        z = torch.randn([self._params['batch_size'], self._z_dim]).to(self.device)  # latent codes
        w = self._generator.mapping(z, None, truncation_psi=0.5, truncation_cutoff=8)
        img = self._generator.synthesis(w, noise_mode='const', force_fp32=True)
        img = img.clamp_(-1, 1)  # .add_(1).div_(2.0)
        # img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        return img

    def classify_gender(self, image: np.ndarray):
        return self._gender(image)

    def get_landmarks(self, image: np.ndarray):
        img, faces, landmarks = self._landmarker.findFaceLandmark(image)
        result = {'image': img, 'faces': faces, 'landmarks': landmarks}
        return result

    def _load_generator(self, path):
        with open(path, 'rb') as f:
            generator = pickle.load(f)['G_ema'].to(self.device)  # torch.nn.Module
        return generator


def gen_images(actor):
    total_images = 400000
    batch_size = 8
    count = 0
    pbar = tqdm(range(total_images // batch_size),
                total=total_images // batch_size,  ## 전체 진행수
                desc='Description',  ## 진행률 앞쪽 출력 문장
                ncols=100,  ## 진행률 출력 폭 조절
                ascii=' =',  ## 바 모양, 첫 번째 문자는 공백이어야 작동
                leave=True,  ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
                )
    root = r'D:\Creadto\Heritage\Dataset\GAN dataset\image'
    for c in pbar:
        begin = time.time()
        images = actor.generate_images()
        gen_time = time.time() - begin
        begin = time.time()
        for i in range(batch_size):
            image = images[i]
            filename = "%06d.jpg" % count
            save_image(image, os.path.join(root, filename))
            count += 1
        save_time = time.time() - begin
        pbar.set_description('Gen images: %.3f sec, Save Images: %.3f sec' % (gen_time, save_time))
    pbar.close()


def label_dataset(root, actor: Actor):
    import pandas as pd
    labels = pd.DataFrame()
    image_list = os.listdir(root)
    count = 0
    with open(r"D:\Creadto\Heritage\Dataset\GAN dataset\label.txt", "a") as f:
        for image_name in image_list[118437:]:
            try:
                print(image_name)
                image = cv2.imread(os.path.join(root, image_name))
                output = actor.classify_gender(image)
                if output.gender_scores[0] < 0.5:
                    gender = 'neutral'
                else:
                    gender = output.genders[0]
                result = actor.get_landmarks(image)
                points = make_3d_points(result['landmarks'])
                df = pd.DataFrame(points[0])
            except:
                os.remove(os.path.join(root, image_name))
                print("Deleted")
                continue

            cv2.imwrite(os.path.join(r"D:\Creadto\Heritage\Dataset\GAN dataset\debug", image_name), result['image'])
            df.to_csv(os.path.join(r"D:\Creadto\Heritage\Dataset\GAN dataset\landmakrs", "%06d.csv" % count),
                      index=False, header=False)
            line = ' '.join(['image/' + image_name, gender, "%.4f" % output.gender_scores[0], "landmark/%06d.csv" % count])
            f.write(line+'\n')
            count += 1
    return labels


def main():
    batch_size = 8
    actor = Actor(generate_path='../../generative_model/stylegan3/pretrained/stylegan3-r-ffhq-1024x1024.pkl',
                  gender_args={'detector_weights': r'../../classification/mivolo/pretrained/yolov8x_person_face.pt',
                               'checkpoint': r'../../classification/mivolo/pretrained/model_imdb_cross_person_4.22_99.46.pth.tar',
                               'device_name': 'cuda:0'},
                  params={'batch_size': batch_size})
    labels = label_dataset(root=r'D:\Creadto\Heritage\Dataset\GAN dataset\image', actor=actor)


if __name__ == "__main__":
    main()

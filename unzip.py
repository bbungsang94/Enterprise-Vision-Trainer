import os
import zipfile


def main(root: str):
    sub_root = root
    folder_stack = []
    while True:
        files: list[str] = os.listdir(sub_root)
        for file in files:
            cur_path = os.path.join(sub_root, file)
            if os.path.isdir(cur_path):
                folder_stack.append(cur_path)
            else:
                if ".zip" in file:
                    with zipfile.ZipFile(cur_path, 'r') as zip_ref:
                        zip_ref.extractall(sub_root)
                    os.remove(cur_path)
                else:
                    print("[" + file + "]Unexpected file extension..: ")
        if len(folder_stack) == 0:
            return
        sub_root = folder_stack.pop()


if __name__ == "__main__":
    dataset_path = r"D:\Creadto\Heritage\Dataset\KAI-hub\169-2.한국인 얼굴 3D 스캐닝 데이터"
    print(main(dataset_path))

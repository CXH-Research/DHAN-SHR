import os
import random
import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, inp='input', target='target', img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, inp)))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, target)))

        self.inp_filenames = [os.path.join(rgb_dir, inp, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, target, x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.transform = A.Compose([
            A.Flip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(p=0.3),
            A.Transpose(p=0.3),
            A.RandomResizedCrop(height=img_options['h'], width=img_options['w']),
        ],
            additional_targets={
                'target': 'image',
            }
        )

    def mixup(self, inp_img, tar_img, mode='mixup'):
        mixup_index_ = random.randint(0, self.sizex - 1)

        mixup_inp_path = self.inp_filenames[mixup_index_]
        mixup_tar_path = self.tar_filenames[mixup_index_]

        mixup_inp_img = Image.open(mixup_inp_path).convert('RGB')
        mixup_tar_img = Image.open(mixup_tar_path).convert('RGB')

        mixup_inp_img = np.array(mixup_inp_img)
        mixup_tar_img = np.array(mixup_tar_img)

        transformed = self.transform(image=mixup_inp_img, target=mixup_tar_img)

        alpha = 0.2
        lam = np.random.beta(alpha, alpha)

        mixup_inp_img = F.to_tensor(transformed['image'])
        mixup_tar_img = F.to_tensor(transformed['target'])

        if mode == 'mixup':
            inp_img = lam * inp_img + (1 - lam) * mixup_inp_img
            tar_img = lam * tar_img + (1 - lam) * mixup_tar_img
        elif mode == 'cutmix':
            img_h, img_w = self.img_options['h'], self.img_options['w']

            cx = np.random.uniform(0, img_w)
            cy = np.random.uniform(0, img_h)

            w = img_w * np.sqrt(1 - lam)
            h = img_h * np.sqrt(1 - lam)

            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, img_w)))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, img_h)))

            inp_img[:, y0:y1, x0:x1] = mixup_inp_img[:, y0:y1, x0:x1]
            tar_img[:, y0:y1, x0:x1] = mixup_tar_img[:, y0:y1, x0:x1]

        return inp_img, tar_img

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = np.array(inp_img)
        tar_img = np.array(tar_img)

        transformed = self.transform(image=inp_img, target=tar_img)

        inp_img = F.to_tensor(transformed['image'])
        tar_img = F.to_tensor(transformed['target'])

        if index_ > 0 and index_ % 3 == 0:
            if random.random() > 0.5:
                inp_img, tar_img = self.mixup(inp_img, tar_img, mode='mixup')
            else:
                inp_img, tar_img = self.mixup(inp_img, tar_img, mode='cutmix')

        filename = os.path.basename(tar_path)

        return inp_img, tar_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, inp='input', target='target', img_options=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, inp)))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, target)))

        self.inp_filenames = [os.path.join(rgb_dir, inp, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, target, x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.transform = A.Compose([
            A.Resize(height=img_options['h'], width=img_options['w']), ],
            additional_targets={
                'target': 'image',
            }
        )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = np.array(inp_img)
        tar_img = np.array(tar_img)

        if not self.img_options['ori']:
            transformed = self.transform(image=inp_img, target=tar_img)

            inp_img = transformed['image']
            tar_img = transformed['target']

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)

        filename = os.path.basename(tar_path)

        return inp_img, tar_img, filename

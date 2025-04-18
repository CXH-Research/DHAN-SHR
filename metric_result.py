import json
import os
import warnings

import albumentations as A
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from config import Config

warnings.filterwarnings('ignore')

opt = Config('config.yml')

result_path = os.path.join(opt.TESTING.RESULT_DIR, opt.MODEL.SESSION)
test_sets = [f for f in os.listdir(result_path)
             if os.path.isdir(os.path.join(result_path, f))]

resize = True
new_size = 256

def get_ref(reference_path, res_fname, h, w):
    tar_fname = os.path.join(reference_path, res_fname)
    alter_suffix = ['jpg', 'JPEG', 'JPG', 'PNG', 'png']
    prefix = tar_fname.split('.')[0]
    i = 0
    while not os.path.exists(tar_fname) and i < len(alter_suffix):
        tar_fname = prefix + '.' + alter_suffix[i]
        i += 1

    tar_img = Image.open(tar_fname).convert('RGB')
    tar_img = to_tensor(tar_img)
    tar_h, tar_w = tar_img.shape[1], tar_img.shape[2]

    if not tar_h == h or not tar_w == w:
        tar_img = tar_img.permute(1, 2, 0).numpy()
        transform = A.Resize(height=h, width=w)
        transformed = transform(image=tar_img)
        tar_img = to_tensor(transformed['image'])
    tar_img = tar_img.cuda()

    return tar_img

def test():
    for benchmark in test_sets:
        reference_path = os.path.join(opt.TESTING.TEST_DIR, benchmark, opt.MODEL.TARGET)

        criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to('cuda:0')

        file_path = os.path.join(result_path, benchmark)
        files = os.listdir(file_path)
        file_num = len(files)

        psnr_sum = 0
        ssim_sum = 0
        lpips_sum = 0

        for filename in files:
            img_file = file_path + '/' + filename

            img = Image.open(img_file).convert('RGB')
            img = to_tensor(img)

            h = img.shape[1]
            w = img.shape[2]
            if resize and not h == w == new_size:
                img = img.permute(1, 2, 0).numpy()
                transform = A.Resize(height=new_size, width=new_size)
                transformed = transform(image=img)
                img = to_tensor(transformed['image'])
                h = w = new_size
            img = img.cuda()

            tar_img = get_ref(reference_path, filename, h, w)

            img = img.unsqueeze(0)
            tar_img = tar_img.unsqueeze(0)
            psnr = peak_signal_noise_ratio(img, tar_img, data_range=1).item()
            ssim = structural_similarity_index_measure(img, tar_img, data_range=1).item()
            lpips = criterion_lpips(img, tar_img).item()

            psnr_sum += psnr
            ssim_sum += ssim
            lpips_sum += lpips

        log_dir = opt.LOG.LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_info = [f'test file path: {file_path}',
                    f'total: {file_num} images']
        avg_psnr = psnr_sum / file_num
        avg_ssim = ssim_sum / file_num
        avg_lpips = lpips_sum / file_num
        log_info.append(f'avg_psnr: {avg_psnr}')
        log_info.append(f'avg_ssim: {avg_ssim}')
        log_info.append(f'avg_lpips: {avg_lpips}')

        with open(os.path.join(opt.LOG.LOG_DIR, opt.TESTING.LOG_FILE), mode='a', encoding='utf-8') as f:
            print(opt.MODEL.SESSION)
            f.write(json.dumps(opt.MODEL.SESSION) + '\n \n')
            for info in log_info:
                print(info)
                f.write(json.dumps(info) + '\n')

            f.write('\n')
            f.close()


if __name__ == '__main__':
    test()

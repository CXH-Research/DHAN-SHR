import cv2
import math
import numpy as np
import kornia.color as color
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch


def uciqe(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # RGB转为HSV
    H, S, V = cv2.split(hsv)
    delta = np.std(H) / 180
    # 色度的标准差
    mu = np.mean(S) / 255  # 饱和度的平均值
    # 求亮度对比值
    n, m = np.shape(V)
    number = math.floor(n * m / 100)
    v = V.flatten() / 255
    v.sort()
    bottom = np.sum(v[:number]) / number
    v = -v
    v.sort()
    v = -v
    top = np.sum(v[:number]) / number
    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    return uciqe


def torch_uciqe(image):
    # RGB转为HSV
    hsv = color.rgb_to_hsv(image)  
    H, S, V = torch.chunk(hsv, 3)

    # 色度的标准差
    delta = torch.std(H) / (2 * math.pi)
    
    # 饱和度的平均值
    mu = torch.mean(S)  
    
    # 求亮度对比值
    n, m = V.shape[1], V.shape[2]
    number = math.floor(n * m / 100)
    v = V.flatten()
    v, _ = v.sort()
    bottom = torch.sum(v[:number]) / number
    v = -v
    v, _ = v.sort()
    v = -v
    top = torch.sum(v[:number]) / number
    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    # uciqe = 0.25 * delta + 0.6 * conl + 0.15 * mu
    return uciqe

def batch_uciqe(images):
    uciqe_sum = 0
    for img in images:
        uciqe = torch_uciqe(img)
        uciqe_sum += uciqe

    avg_ucique = uciqe_sum / len(images)
    return avg_ucique

if __name__ == '__main__':
    image = '../result/EUVP-1/test_p0_.jpg'
    img = Image.open(image).convert('RGB')
    img_tensor = to_tensor(img).cuda()
    img = np.array(img)
    # img = torch.cat((img, img), 0)
    # print(img.shape)
    print(uciqe(img))
    print(torch_uciqe(img_tensor))


    
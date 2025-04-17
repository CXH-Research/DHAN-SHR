from PIL import Image
import torch
import torch.nn.functional as F

from torchvision.transforms.functional import to_tensor
import math


sobel_kernel_x = torch.tensor(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
sobel_kernel_y = torch.tensor(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)


def _uiconm(x, window_size):
    # Ensure image is divisible by window_size - doesn't matter if we cut out some pixels
    k1 = x.shape[2] // window_size
    k2 = x.shape[1] // window_size
    x = x[:, :k2 * window_size, :k1 * window_size]

    # Weight
    w = -1. / (k1 * k2)

    # Entropy scale - higher helps with randomness
    alpha = 1

    # Create blocks
    # 3, 108, 192, 10, 10
    x = x.unfold(1, window_size, window_size).unfold(
        2, window_size, window_size)
    x = x.reshape(-1, k2, k1, window_size, window_size)

    # Compute min and max values for each block
    min_ = torch.min(torch.min(torch.min(x, dim=-1).values,
                     dim=-1).values, dim=0).values
    max_ = torch.max(torch.max(torch.max(x, dim=-1).values,
                     dim=-1).values, dim=0).values

    # Calculate top and bot
    top = max_ - min_
    bot = max_ + min_

    # Calculate the value for each block
    val = alpha * torch.pow((top / bot), alpha) * torch.log(top / bot)

    # Handle NaN and zero values
    val = torch.where(torch.isnan(val) | (bot == 0.0) |
                      (top == 0.0), torch.zeros_like(val), val)

    # Sum up the values and apply the weight
    val = w * val.sum()

    return val


def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = x.sort()[0]

    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)

    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s = int(T_a_L+1)
    e = int(K-T_a_R)
    val = torch.sum(x[s:e])
    val = weight*val
    return val


def s_a(x, mu):
    val = torch.sum(torch.pow(x - mu, 2)) / len(x)
    return val


def _uicm(x):
    R = x[0, :, :].flatten()
    G = x[1, :, :].flatten()
    B = x[2, :, :].flatten()
    RG = R-G
    YB = ((R+G)/2)-B

    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)

    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)

    l = torch.sqrt((torch.pow(mu_a_RG, 2)+torch.pow(mu_a_YB, 2)))
    r = torch.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)


def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[0, :, :]
    G = x[1, :, :]
    B = x[2, :, :]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel_torch(R)
    Gs = sobel_torch(G)
    Bs = sobel_torch(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = torch.multiply(Rs, R)
    G_edge_map = torch.multiply(Gs, G)
    B_edge_map = torch.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)


def eme(x, window_size):
    """
    Enhancement measure estimation
    x.shape[0] = height
    x.shape[1] = width
    """
    # Ensure image is divisible by window_size - doesn't matter if we cut out some pixels
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    x = x[:k2 * window_size, :k1 * window_size]

    # Reshape x into a tensor with shape (k2, window_size, k1, window_size)
    x = x.view(k2, window_size, k1, window_size)

    # Transpose and reshape the tensor into shape (k2*k1, window_size*window_size)
    x = x.permute(0, 2, 1, 3).contiguous().view(-1, window_size * window_size)

    # Compute the max and min values for each block
    max_vals, _ = torch.max(x, dim=1)
    min_vals, _ = torch.min(x, dim=1)

    # Bound checks, can't do log(0)
    non_zero_mask = (min_vals != 0) & (max_vals != 0)

    # Compute the log ratios
    log_ratios = torch.zeros_like(max_vals)
    log_ratios[non_zero_mask] = torch.log(
        max_vals[non_zero_mask] / min_vals[non_zero_mask])

    # Compute the sum of the log ratios
    val = log_ratios.sum()

    # Compute the weight
    w = 2. / (k1 * k2)

    return w * val


def sobel_torch(x):
    x = x.squeeze(0)
    dx = F.conv2d(x[None, None], sobel_kernel_x.to(x.device), padding=1)
    dy = F.conv2d(x[None, None], sobel_kernel_y.to(x.device), padding=1)
    mag = torch.hypot(dx, dy)
    mag *= 255.0 / torch.max(mag)
    return mag.squeeze()

def torch_uiqm(img):

    # x = img.mul(255).add_(0.5).clamp_(0, 255)
    x = img * 255

    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    # c1 = c2 = c3 = 0.3333

    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 10)

    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)

    return uiqm

def batch_uiqm(images):
    uiqm_sum = 0
    for img in images:
        uiqm = torch_uiqm(img)
        uiqm_sum += uiqm
    avg_uiqm = uiqm_sum / len(images)

    return avg_uiqm


if __name__ == '__main__':
    x = Image.open('../result/EUVP-1/test_p0_.jpg').convert('RGB')
    x = to_tensor(x).cuda().unsqueeze(0)
    x = torch.cat((x, x, x, x), 0)
    print(batch_uiqm(x))


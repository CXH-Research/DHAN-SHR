import json
import warnings

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_validation_data
from models import *
from utils import *

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)


def test():
    accelerator = Accelerator()

    # Data Loader
    val_dir = opt.TESTING.TEST_DIR

    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

    model = Model()

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model = accelerator.prepare(model)

    if not os.path.exists(opt.LOG.LOG_DIR):
        os.makedirs(opt.LOG.LOG_DIR)

    for entry in os.scandir(val_dir):
        if entry.is_dir():
            testset_name = os.path.basename(entry)
            result_dir = os.path.join(opt.TESTING.RESULT_DIR, opt.MODEL.SESSION, testset_name)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            test_one_set(accelerator, model, opt, entry, criterion_lpips, result_dir)


def test_one_set(accelerator, model, opt, val_dir, criterion_lpips, result_dir):

    val_dataset = get_validation_data(val_dir, opt.MODEL.INPUT, opt.MODEL.TARGET,
                                      {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': opt.TESTING.ORI})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    testloader = accelerator.prepare(testloader)

    model.eval()

    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_lpips = 0
    for _, test_data in enumerate(tqdm(testloader)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        tar = test_data[1]

        with torch.no_grad():
            res = model(inp).clamp(0, 1)

        save_image(res, os.path.join(result_dir, test_data[2][0]))

        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1)
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1)
        stat_lpips += criterion_lpips(res, tar).item()

    stat_psnr /= size
    stat_ssim /= size
    stat_lpips /= size

    test_info = ("Test Result on {}, check point {}, testing data {}".
                 format(opt.MODEL.SESSION, opt.TESTING.WEIGHT, os.path.basename(val_dir)))
    log_stats = ("PSNR: {}, SSIM: {}, LPIPS: {}".format(stat_psnr, stat_ssim, stat_lpips))
    print(test_info)
    print(log_stats)

    with open(os.path.join(opt.LOG.LOG_DIR, opt.TESTING.LOG_FILE), mode='a', encoding='utf-8') as f:
        f.write(json.dumps(test_info) + '\n')
        f.write(json.dumps(log_stats) + '\n')


if __name__ == '__main__':
    test()

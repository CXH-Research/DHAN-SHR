import json
import time
import warnings

import torch.optim as optim
from accelerate import Accelerator

from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_absolute_error
from tqdm import tqdm

from config import Config
from data import get_training_data, get_validation_data
from models import *
from utils import *

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)

def train():
    # Accelerate
    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
    device = accelerator.device
    config = {
        "dataset": opt.TRAINING.TRAIN_DIR
    }
    accelerator.init_trackers("Highlight", config=config)

    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)

    # Data Loader
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    train_dataset = get_training_data(train_dir, opt.MODEL.INPUT, opt.MODEL.TARGET,
                                      {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                             drop_last=False, pin_memory=True)
    val_dataset = get_validation_data(val_dir, opt.MODEL.INPUT, opt.MODEL.TARGET,
                                      {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': opt.TRAINING.ORI})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Loss
    model = Model()

    # criterion_ssim = SSIM(data_range=1, size_average=True, channel=3).to(device)
    criterion_psnr = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer_b = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.OPTIM.LR_INITIAL,
                              betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    start_epoch = 1

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    best_epoch = 1
    best_psnr = 0
    size = len(testloader)
    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()
        for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            inp = data[0].contiguous()
            tar = data[1]

            # forward
            optimizer_b.zero_grad()
            res = model(inp)

            loss_psnr = criterion_psnr(res, tar)
            loss_ssim = 1 - structural_similarity_index_measure(res, tar, data_range=1)

            train_loss = loss_psnr + 0.4 * loss_ssim

            # backward
            accelerator.backward(train_loss)
            optimizer_b.step()

        scheduler_b.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr = 0
            ssim = 0
            mae = 0
            for _, test_data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                # get the inputs; data is a list of [targets, inputs, filename]
                inp = test_data[0].contiguous()
                tar = test_data[1]

                with torch.no_grad():
                    res = model(inp)

                res, tar = accelerator.gather((res, tar))

                psnr += peak_signal_noise_ratio(res, tar, data_range=1)
                ssim += structural_similarity_index_measure(res, tar, data_range=1)
                mae += mean_absolute_error(torch.mul(res, 255), torch.mul(tar, 255))

            psnr /= size
            ssim /= size
            mae /= size

            if psnr > best_psnr:
                # save model
                best_epoch = epoch
                best_psnr = psnr
                save_checkpoint({
                    'state_dict': model.state_dict(),
                }, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)

            accelerator.log({
                "PSNR": psnr,
                "SSIM": ssim,
                "MAE": mae
            }, step=epoch)

            log_stats = ("epoch: {}, PSNR: {}, SSIM: {}, MAE: {}, best PSNR: {}, best epoch: {}"
                .format(epoch, psnr, ssim, mae, best_psnr, best_epoch))
            print(log_stats)
            with open(os.path.join(opt.LOG.LOG_DIR, opt.TRAINING.LOG_FILE), mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

    accelerator.end_training()


if __name__ == '__main__':
    train()

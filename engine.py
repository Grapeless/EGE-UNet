import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from scipy.ndimage import distance_transform_edt, binary_erosion
from utils import save_imgs


def _extract_border(mask):
    """Extract border pixels via morphological erosion."""
    mask_bool = mask.astype(bool)
    eroded = binary_erosion(mask_bool, structure=np.ones((3, 3)))
    border = mask_bool ^ eroded  # XOR: border = mask minus eroded interior
    if border.sum() == 0:
        return mask_bool  # fallback for single-pixel regions
    return border


def compute_hd95(pred, gt):
    """Compute 95th percentile Hausdorff Distance between binary mask borders.

    Args:
        pred: binary numpy array (H, W)
        gt: binary numpy array (H, W)
    Returns:
        hd95 value (float), or np.inf if either mask is empty
    """
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf

    pred_border = _extract_border(pred)
    gt_border = _extract_border(gt)

    # Distance from pred border to nearest gt pixel
    gt_dist = distance_transform_edt(~gt.astype(bool))
    d_pred_to_gt = gt_dist[pred_border]

    # Distance from gt border to nearest pred pixel
    pred_dist = distance_transform_edt(~pred.astype(bool))
    d_gt_to_pred = pred_dist[gt_border]

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return np.percentile(all_distances, 95)


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        # Mixup augmentation (30% probability)
        use_mixup = getattr(config, 'use_mixup', False)
        if use_mixup and random.random() < 0.3:
            lam = np.random.beta(0.2, 0.2)
            index = torch.randperm(images.size(0)).cuda()
            images = lam * images + (1 - lam) * images[index]
            targets = lam * targets + (1 - lam) * targets[index]

        gt_pre, out, boundary_out = model(images)
        if boundary_out is not None:
            loss = criterion(gt_pre, out, targets, boundary_logits=boundary_out)
        else:
            loss = criterion(gt_pre, out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre, out, boundary_out = model(img)
            if boundary_out is not None:
                loss = criterion(gt_pre, out, msk, boundary_logits=boundary_out)
            else:
                loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None,
                    use_tta=False):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    hd95_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            if use_tta:
                # TTA: average predictions over 4 flips
                out = _tta_predict(model, img)
                gt_pre = None
                boundary_out = None
                loss = torch.tensor(0.0)  # skip loss for TTA
            else:
                gt_pre, out, boundary_out = model(img)
                if boundary_out is not None:
                    loss = criterion(gt_pre, out, msk, boundary_logits=boundary_out)
                else:
                    loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            msk_np = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk_np)
            if type(out) is tuple:
                out = out[0]
            out_np = out.squeeze(1).cpu().detach().numpy()
            preds.append(out_np)

            # Per-image HD95
            for b in range(out_np.shape[0] if out_np.ndim == 3 else 1):
                pred_b = out_np[b] if out_np.ndim == 3 else out_np
                gt_b = msk_np[b] if msk_np.ndim == 3 else msk_np
                pred_bin = (pred_b >= config.threshold).astype(np.uint8)
                gt_bin = (gt_b >= 0.5).astype(np.uint8)
                hd = compute_hd95(pred_bin, gt_bin)
                if hd != np.inf:
                    hd95_list.append(hd)

            if i % config.save_interval == 0:
                save_imgs(img, msk_np, out_np, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        hd95_mean = np.mean(hd95_list) if hd95_list else float('inf')

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        tta_tag = ' [TTA]' if use_tta else ''
        log_info = f'test of best model{tta_tag}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, hd95: {hd95_mean:.2f}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def _tta_predict(model, img):
    """Test-Time Augmentation: average over original + 3 flips."""
    preds = []

    # Original
    _, out, _ = model(img)
    if type(out) is tuple:
        out = out[0]
    preds.append(out)

    # Horizontal flip
    img_h = torch.flip(img, [3])
    _, out_h, _ = model(img_h)
    if type(out_h) is tuple:
        out_h = out_h[0]
    preds.append(torch.flip(out_h, [3]))

    # Vertical flip
    img_v = torch.flip(img, [2])
    _, out_v, _ = model(img_v)
    if type(out_v) is tuple:
        out_v = out_v[0]
    preds.append(torch.flip(out_v, [2]))

    # Both flips
    img_hv = torch.flip(img, [2, 3])
    _, out_hv, _ = model(img_hv)
    if type(out_hv) is tuple:
        out_hv = out_hv[0]
    preds.append(torch.flip(out_hv, [2, 3]))

    return torch.stack(preds).mean(dim=0)

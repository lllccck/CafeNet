import torch.nn as nn
import torch.nn.functional as F
import torch

"""BCE loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred, targets, smooth=1):
        
        #comment out if your models contains a sigmoid or equivalent activation layer
        #pred = F.sigmoid(pred)       
        
        #flatten label and prediction tensors
        pred = pred.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (pred * targets).sum()
        total = (pred + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, reduction)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss




"""BCE + IoU Loss"""


class BceIoULoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BceIoULoss, self).__init__()
        self.bce = BCELoss(weight, reduction)
        self.iou = IoULoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        iouloss = self.iou(pred, target)

        loss = iouloss + bceloss

        return loss

""" Structure Loss: https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py """
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()
    

""" Deep Supervision Loss"""


def DeepSupervisionLoss(pred, gt,typeloss="BceDiceLoss"):
    d0, d1, d2, d3 = pred[0:]

    if typeloss=="BceDiceLoss":
      criterion = BceDiceLoss()
    elif typeloss=="BceIoULoss":
      criterion = BceIoULoss()
    elif typeloss=="StructureLoss":
      criterion = StructureLoss()
    else:
      raise Exception("Loss name is unvalid.")

    loss0 = criterion(d0, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss1 = criterion(d1, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion(d2, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion(d3, gt)

    return loss0 + loss1 + loss2 + loss3

def DeepStructureLoss(pred, gt):
    d0, d1, d2, d3 = pred[0:]
    criterion = StructureLoss()
    loss0 = criterion(d0, gt)
    loss1 = criterion(d1, gt)
    loss2 = criterion(d2, gt)
    loss3 = criterion(d3, gt)

    return loss0 + loss1 + loss2 + loss3


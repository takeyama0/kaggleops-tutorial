import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda")



def dice_loss(pred, target, smooth=1e-10):
    intersection = (torch.sigmoid(pred) * target).sum(axis=1, keepdim=True)
    s_pred   = pred.sum(axis=1, keepdim=True)
    s_target = target.sum(axis=1, keepdim=True)
    loss = 1 - ((2. * intersection + smooth) / (s_pred + s_target + smooth))
    return loss

def jaccard_loss(pred, target, smooth=1e-10):
    intersection = (torch.sigmoid(pred) * target).sum(axis=1, keepdim=True)
    s_pred   = pred.sum(axis=1, keepdim=True)
    s_target = target.sum(axis=1, keepdim=True)
    loss = 1 - ((intersection + smooth) / (s_pred + s_target - intersection + smooth))
    return loss


class BCE_JACCARD:
    def __init__(self, BCE_ratio=0.5, smooth=1e-10):
        self.BCE_ratio = BCE_ratio
        self.smooth    = smooth
        self.BCELoss   = nn.BCEWithLogitsLoss(weight=None, reduction='none')

    def __call__(self, pred, target):
        _bce  = self.BCELoss(pred, target)
        _jacc = jaccard_loss(pred, target, smooth=self.smooth)
        return self.BCE_ratio * _bce + (1 - self.BCE_ratio) * _jacc


class BCE_DICE:
    def __init__(self, BCE_ratio=0.5, smooth=1e-10):
        self.BCE_ratio = BCE_ratio
        self.smooth    = smooth
        self.BCELoss   = nn.BCEWithLogitsLoss(weight=None, reduction='none')

    def __call__(self, pred, target):
        _bce  = self.BCELoss(pred, target)
        _dice = dice_loss(pred, target, smooth=self.smooth)
        return self.BCE_ratio * _bce + (1 - self.BCE_ratio) * _dice


class CROSS_ENTROPY:
    def __init__(self):
        self.CELoss = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, pred, target):
        s_pre, e_pre = pred.split(1, dim=-1)
        s_tar, e_tar = target.split(1, dim=-1)

        s_loss = self.CELoss(s_pre.squeeze(-1), s_tar.squeeze(-1))
        e_loss = self.CELoss(e_pre.squeeze(-1), e_tar.squeeze(-1))

        loss = (s_loss + e_loss) / 2
        return loss


class SPAN_SEGMENT:
    def __init__(self, BCE_ratio=0.5, SPAN_ratio = 1.0):
        self.bce_jaccard   = BCE_JACCARD(BCE_ratio=BCE_ratio, smooth=1e-10)
        self.cross_entropy = CROSS_ENTROPY()
        self.span_ratio    = SPAN_ratio
        
    def __call__(self, pred, target):
        loss_ce = self.cross_entropy(pred[:,:,:2], target['target_span'])
        loss_ce = loss_ce * target['weight']
        loss_ce = loss_ce.view(-1,1)

        loss_bj = self.bce_jaccard(pred[:,:,2:].squeeze(-1), target['target'])
        loss_bj = loss_bj / (target['mask'].sum(axis=1, keepdim=True)-2.0)
        loss_bj = loss_bj * target['weight'].view(-1,1)

        loss = self.span_ratio * loss_ce + (1 - self.span_ratio) * loss_bj
        return loss
        

class LOVASZ_HINGE:
    def __init__(self, per_image=True, ignore=None):
        self.per_image = per_image
        self.ignore = ignore
    
    def __call__(self, out, labels):
        return L.lovasz_hinge(out.unsqueeze(-1), labels.unsqueeze(-1), per_image=self.per_image, ignore=self.ignore)

class CROSS_ENTROPY_2_1:
    def __init__(self):
        self.CELoss = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, pred, target):
        s_pre, e_pre = pred.split(1, dim=-1)
        s_tar, e_tar = target.split(1, dim=-1)

        s_loss = self.CELoss(s_pre.squeeze(-1), s_tar.squeeze(-1))
        e_loss = self.CELoss(e_pre.squeeze(-1), e_tar.squeeze(-1))

        s_weight = 2

        loss = ((s_weight/(s_weight+1))*s_loss + (1/(s_weight+1))*e_loss) / 2
        return loss



class CROSS_ENTROPY_DIST_WEIGHT:
    def __init__(self):
        self.CELoss = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, pred, target):
        s_pre, e_pre = pred.split(1, dim=-1)
        s_tar, e_tar = target.split(1, dim=-1)

        s_loss = self.CELoss(s_pre.squeeze(-1), s_tar.squeeze(-1))
        e_loss = self.CELoss(e_pre.squeeze(-1), e_tar.squeeze(-1))


        token_length = pred.detach().cpu().numpy()[:,:,0]
        token_length = (token_length!=-1e10).sum(axis=1) # (N)
        token_length = np.repeat(token_length, 2).reshape(-1,2) # (N, 2)
        pred_idx     = pred.detach().cpu().numpy().argmax(axis=1) # (N, 2)
        targ_idx     = target.detach().cpu().numpy() # (N, 2)
        diff_dist    = np.abs(pred_idx - targ_idx) / token_length  # (N, 2)
        diff_dist    = diff_dist**2

        diff_dist = torch.tensor((diff_dist-diff_dist.mean() + 1), dtype=torch.float32).to(DEVICE)

        loss = (s_loss*diff_dist[:,0] + e_loss*diff_dist[:,1]) / 2
        return loss

def define_criterion(CONFIG):
    if   CONFIG['LOSS_TYPE']=='BCE':
        criterion = nn.BCEWithLogitsLoss(weight=None, reduction='none')

    elif CONFIG['LOSS_TYPE']=='BCE_DICE':
        criterion = BCE_DICE(BCE_ratio=CONFIG['BCE_RATIO'], smooth=1e-10)

    elif CONFIG['LOSS_TYPE']=='BCE_JACCARD':
        criterion = BCE_JACCARD(BCE_ratio=CONFIG['BCE_RATIO'], smooth=1e-10)

    elif CONFIG['LOSS_TYPE']=='LOVASZ_HINGE': 
        criterion = LOVASZ_HINGE()

    elif CONFIG['LOSS_TYPE']=='CROSS_ENTROPY':
        criterion = CROSS_ENTROPY()

    elif CONFIG["LOSS_TYPE"]=='SPAN_SEGMENT':
        criterion = SPAN_SEGMENT(BCE_ratio=0.5, SPAN_ratio=CONFIG['SPAN_RATIO'])

    elif CONFIG['LOSS_TYPE']=='CROSS_ENTROPY_2_1':
        criterion = CROSS_ENTROPY_2_1()

    elif CONFIG['LOSS_TYPE']=='CROSS_ENTROPY_DIST_WEIGHT':
        criterion = CROSS_ENTROPY_DIST_WEIGHT()
        

    return criterion
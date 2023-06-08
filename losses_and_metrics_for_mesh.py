import torch
import numpy as np

def weighting_DSC(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    mdsc = 0.0
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        mdsc += w*((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))
        
    return mdsc


def weighting_SEN(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    msen = 0.0
    n_classes = y_pred.shape[-1] 

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        msen += w*((intersection + smooth) / (true_flat.sum() + smooth))
        
    return msen


def weighting_PPV(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    mppv = 0.0
    n_classes = y_pred.shape[-1] 

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        mppv += w*((intersection + smooth) / (pred_flat.sum() + smooth))
        
    return mppv

   
def Generalized_Dice_Loss(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    loss = 0.
    n_classes = y_pred.shape[-1]
    
    for c in range(0, n_classes):
        pred_flat = y_pred[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
       
        # with weight
        w = class_weights[c]/class_weights.sum()
        loss += w*(1 - ((2. * intersection + smooth) /
                         (pred_flat.sum() + true_flat.sum() + smooth)))
       
    return loss


def DiceCELoss(y_pred, y_true, class_weights, diceloss_ratio=0.5):
    """
    dice loss 和 cross entropy loss 的结合
    """
    assert 0 <= diceloss_ratio <= 1
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')(y_pred, y_true.float())
    dice_loss = Generalized_Dice_Loss(y_pred, y_true, class_weights)

    return ce_loss * (1 - diceloss_ratio) + dice_loss * diceloss_ratio

def CELoss(y_pred, y_true):
    ce = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = ce(y_pred, y_true.float())
    loss = loss / y_true.shape[1]
    return loss


def DSC(y_pred, y_true, ignore_background=False, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    dsc = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
        
    return dsc


def SEN(y_pred, y_true, ignore_background=False, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    sen = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))
            
        sen = np.asarray(sen)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))
            
        sen = np.asarray(sen)
        
    return sen


def PPV(y_pred, y_true, ignore_background=False, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    ppv = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))
            
        ppv = np.asarray(ppv)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))
            
        ppv = np.asarray(ppv)
        
    return ppv

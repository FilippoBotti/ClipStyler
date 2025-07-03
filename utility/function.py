import torch
import clip
import math
from utility.template import imagenet_templates
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_mean_std1(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # assert (len(size) == 4)
    WH,N, C = size
    feat_var = feat.var(dim=0) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.mean(dim=0)
    return feat_mean, feat_std
def normal(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized 
def normal_style(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std1(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())


def truncate(f):
    n=3
    return math.trunc(f * 10**n) / 10**n

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

def clip_normalize(image, device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def get_clip_score(clip_model, img, text, device):
    #out1, out2 = clip_model.encode(clip_normalize(img), clip.tokenize(args.text).to(device))
    img_feature = clip_model.encode_image(clip_normalize(img, device))
    tx_feature = clip_model.encode_text(clip.tokenize(text).to(device))
    img_feature = img_feature / img_feature.norm(dim=1, keepdim=True)
    tx_feature = tx_feature / tx_feature.norm(dim=1, keepdim=True)
    score = img_feature @ tx_feature.t()
    
    score = score.mean()
    #print("clip score " + str(score.item()))
    return score

def get_ssim(preds, target, device):
    ssim = StructuralSimilarityIndexMeasure().to(device)
    score = ssim(preds, target)
    #print("SSIM " + str(score.item()))
    return score

def get_aesthetic_score(clip_model, model_ae, img, device):
    img_emb = clip_model.encode_image(clip_normalize(img, device))
    score = model_ae(img_emb)
    score = score.mean()
    #print("Aesthetic score " + str(score.item()))
    return score


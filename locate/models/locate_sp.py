# supervised version of LOCATE
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dino import vision_transformer as vits
from models.dino.utils import load_pretrained_weights
from models.model_util import *
from fast_pytorch_kmeans import KMeans
from .locate import Mlp

import pdb


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    #pdb.set_trace()

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean()
    # return loss.mean(1).sum() / num_boxes



class Net(nn.Module):

    def __init__(self, aff_classes=36):
        super(Net, self).__init__()

        self.aff_classes = aff_classes
        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- hyper-parameters --- #
        self.aff_cam_thd = 0.6
        self.part_iou_thd = 0.6
        self.cel_margin = 0.5

        # --- dino-vit features --- #
        self.vit_feat_dim = 384
        self.cluster_num = 3
        self.stride = 16
        self.patch = 16

        self.vit_model = vits.__dict__['vit_small'](patch_size=self.patch, num_classes=0)
        load_pretrained_weights(self.vit_model, '', None, 'vit_small', self.patch)

        # --- learning parameters --- #
        self.aff_proj = Mlp(in_features=self.vit_feat_dim, hidden_features=int(self.vit_feat_dim * 4),
                            act_layer=nn.GELU, drop=0.)
        self.aff_ego_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_exo_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_fc = nn.Conv2d(self.vit_feat_dim, self.aff_classes, 1)

    def forward(self, ego, aff_label, aff_gt, epoch):
        _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # attn: b x 6 x (1+hw) x (1+hw)
        ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1)
        ego_proj = ego_desc[:, 1:] + self.aff_proj(ego_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        b, c, h, w = ego_desc.shape
        ego_proj = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]

        gt_ego_cam = gt_ego_cam.unsqueeze(1)

        # gt_ego_cam = F.interpolate(gt_ego_cam, (224, 224))
        aff_gt = F.interpolate(aff_gt, (14, 14))

        # (B, N_affordance, 14, 14)
        loss_aff = sigmoid_focal_loss(
            gt_ego_cam,
            aff_gt,
            aff_gt.shape[0],
            #alpha=0.95,
        )

        # loss_aff = F.binary_cross_entropy_with_logits(gt_ego_cam, aff_gt)

        # breakpoint()


        # masks = {'exo_aff': gt_aff_cam, 'ego_sam': ego_sam,
        #          'pred': (sim_maps, exo_sim_maps, part_score, gt_ego_cam)}
        # logits = {'aff': aff_logits, 'aff_ego': aff_logits_ego}

        return loss_aff

    @torch.no_grad()
    def test_forward(self, ego, aff_label):
        _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # attn: b x 6 x (1+hw) x (1+hw)
        ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1)
        ego_proj = ego_desc[:, 1:] + self.aff_proj(ego_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        b, c, h, w = ego_desc.shape
        ego_proj = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]

        gt_ego_cam = gt_ego_cam.sigmoid()

        return gt_ego_cam

    def _reshape_transform(self, tensor, patch_size, stride):
        height = (224 - patch_size) // stride + 1
        width = (224 - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.transpose(2, 3).transpose(1, 2).contiguous()
        return result

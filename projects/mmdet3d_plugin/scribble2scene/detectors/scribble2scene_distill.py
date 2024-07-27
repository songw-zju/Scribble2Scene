import time
import copy
import torch
import numpy as np
import mmdet3d
from tkinter.messagebox import NO
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .scribble2scene import Scribble2Scene
from mmcv.runner import load_checkpoint
from mmcv.runner import BaseModule
import torch.nn as nn


# Teacher Model
class ScribbleWSTeacherOccWrapper(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.teacher = teacher
        self.load_pre_checkpoint('./ckpts/miou21.7_iou82.8_epoch_9.pth')
        for name, para in self.teacher.named_parameters():
            para.requires_grad_(False)

    @torch.no_grad()
    def load_pre_checkpoint(self, checkpoint_path):
        load_checkpoint(self.teacher, checkpoint_path, map_location='cpu', strict=False)


class ScribbleWSTeacherOcc(Scribble2Scene):

    @torch.no_grad()
    def momentum_update(self, cur_model, teacher_momentum=None):
        """
        Momentum update of the key encoder
        """
        if teacher_momentum is None:
            teacher_momentum = self.teacher_momentum
        for param_q, param_k in zip(cur_model.parameters(), self.parameters()):
            param_k.data = param_k.data * teacher_momentum + param_q.data * (1. - teacher_momentum)


@DETECTORS.register_module()
class Scribble2Scene_Distill(Scribble2Scene):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):

        super(Scribble2Scene_Distill,
              self).__init__(use_grid_mask, pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        self.tsw = ScribbleWSTeacherOccWrapper(
            ScribbleWSTeacherOcc(use_grid_mask, pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained))
        self.tsw.teacher.eval()

    def forward_pts_train(self,
                          img_feats,
                          teacher_img_feats,
                          img,
                          img_metas,
                          target,
                          scribble,
                          pseudo):
        """Forward function'
        """
        outs, stu_feat = self.pts_bbox_head(img_feats, img_metas, target, scribble, pseudo)
        teacher_outs, tea_feat = self.tsw.teacher.pts_bbox_head(teacher_img_feats, img_metas, target, scribble, pseudo, use_teacher=True)
        # print(outs)
        losses = self.pts_bbox_head.training_step(outs, teacher_outs, stu_feat, tea_feat, target, scribble, pseudo, img, img_metas)
        return losses

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      img=None,
                      target=None,
                      scribble=None,
                      pseudo=None):
        """Forward training function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]

        img_metas = [each[len_queue - 1] for each in img_metas]
        img = img[:, -1, ...]
        img_feats = self.extract_feat(img=img)
        teacher_img_feats = self.tsw.teacher.extract_feat(img=img)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, teacher_img_feats, img, img_metas, target, scribble, pseudo)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     img_metas=None,
                     img=None,
                     target=None,
                     scribble=None,
                     pseudo=None,
                     **kwargs):
        """Forward testing function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Completion result.
        """

        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]

        img_metas = [each[len_queue - 1] for each in img_metas]
        img = img[:, -1, ...]
        img_feats = self.extract_feat(img=img)
        # img_feats = self.tsw.teacher.extract_feat(img=img)
        outs, _ = self.pts_bbox_head(img_feats, img_metas, target, scribble, pseudo)
        # outs = self.tsw.teacher.pts_bbox_head(img_feats, img_metas, target, scribble, pseudo, use_teacher=True)
        completion_results = self.pts_bbox_head.validation_step(outs, target, scribble, pseudo, img_metas)

        return completion_results

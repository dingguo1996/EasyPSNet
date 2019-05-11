import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import seg_cross_entropy


@HEADS.register_module
class FCNSegHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='bilinear',
                 upsample_ratio=2,
                 num_classes=133,
                 class_agnostic=False,
                 normalize=None):
        super(FCNSegHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = 133#dinggguo
        self.class_agnostic = class_agnostic
        self.normalize = normalize
        self.with_bias = normalize is None

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(self.num_convs):
            in_channels = self.in_channels
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    normalize=normalize,
                    bias=self.with_bias))
        if self.upsample_method is None:
            self.upsample = False
            self.upsamples = None
        elif self.upsample_method == 'deconv':
            self.upsample = True
            for i in range(self.num_convs):
                self.upsamples .append(
                        nn.ConvTranspose2d(
                        self.conv_out_channels,
                        self.conv_out_channels,
                        self.upsample_ratio**i,
                        stride=self.upsample_ratio**i)
                )
        else:
            self.upsample = True
            self.upsamples = None

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsamples, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv_feats=[]
        upsample_feats=[]
        for i, conv in enumerate(self.convs):
            conv_feats.append(conv(x[i]))
        if self.upsample is True and self.upsamples is not None and self.upsample_method == 'deconv':
            for i, upsample in enumerate(self.upsamples):
                if self.upsample_method == 'deconv':
                    upsample_feats.append(upsample(conv_feats[i]))
                    upsample_feats[i] = self.relu(upsample_feats[i])
        elif self.upsample is True and self.upsamples is None:
            for i in range(self.num_convs):
                upsample_feats.append(F.interpolate(conv_feats[i],
                                        scale_factor=self.upsample_ratio**i,mode=self.upsample_method))
        else:
            return conv_feats

        for i, upsample_feat in enumerate(upsample_feats):
            if i ==0:
                seg_feat=upsample_feats[0]
            else:
                seg_feat=seg_feat+upsample_feats[i]
        seg_feat=F.interpolate(seg_feat,scale_factor=4,mode='bilinear')
        seg_pred = self.conv_logits(seg_feat)
        return seg_pred

    def loss(self, seg_pred, gt_seg):
        loss = dict()
        if self.class_agnostic:
            loss_seg = seg_cross_entropy(seg_pred, gt_seg)
        else:
            loss_seg = seg_cross_entropy(seg_pred, gt_seg)
        loss['loss_seg'] = loss_seg
        return loss

    def get_seg_masks(self, seg_pred, rcnn_test_cfg, ori_shape,
                      img_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(seg_pred, torch.Tensor):
            seg_pred = seg_pred[...,:img_shape[0],:img_shape[1]].softmax(dim=1).cpu().numpy()
            seg_pred_cls = seg_pred.argmax(axis=1)
        assert isinstance(seg_pred, np.ndarray)
        assert isinstance(seg_pred_cls, np.ndarray)


        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        seg_image=mmcv.imresize(seg_pred_cls.transpose(1,2,0), (img_w, img_h), interpolation='nearest')

        return seg_image


from .two_stage_panotic import TwoStagePanoticDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class PanoticRCNN(TwoStagePanoticDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 seg_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(PanoticRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            seg_head=seg_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

# -*- coding: utf-8 -*-
import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]  B * 8732 * 4
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes] B * 8732 * 21
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)  # B * 21 * top_k * 5
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):  # i是batch id
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)  # 8732 * 4， decode_boxes是以(left, top, right, bottom)的形式出现
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):  # nms是对每类做的
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # c_mask:（300,）,假设某类的>conf_thresh的有300个idx,
                scores = conf_scores[cl][c_mask]  # (300,)，某类的8732个分数
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)  # 300 * 4
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k) # 300 * 4 -> ids: (300,), count: 一个数，表示最后框的数量。ids中的元素只有count个是有效的
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1), #  B * 21 * top_k * (1 + 4)，但只有B * 21 * count * 5是有效的
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5) # ???
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

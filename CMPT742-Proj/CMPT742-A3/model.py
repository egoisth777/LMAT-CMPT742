import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F


def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    # input:
    # pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    # output:
    # loss -- a single number for the value of the loss function, [1]

    # TODO: write a loss function for SSD
    #
    # For confidence (class labels), use cross entropy (F.cross_entropy)
    # You can try F.binary_cross_entropy and see which loss is better
    # For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    # Note that you need to consider cells carrying objects and empty cells separately.
    # I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    # and reshape box to [batch_size*num_of_boxes, 4].
    num_of_classes = pred_confidence.size(dim=2)

    pred_confidence = torch.reshape(pred_confidence, (-1, num_of_classes))
    pred_box = torch.reshape(pred_box, (-1, 4))
    ann_confidence = torch.reshape(ann_confidence, (-1, num_of_classes))
    ann_box = torch.reshape(ann_box, (-1, 4))
    # Then you need to figure out how you can get the indices of all cells carrying objects,
    # and use confidence[indices], box[indices] to select those cells.
    obj_mask = ann_confidence[:, :-1].sum(dim=1) > 0
    noobj_mask = ~obj_mask
    obj_conf_loss = F.cross_entropy(
        pred_confidence[obj_mask], ann_confidence[obj_mask].argmax(dim=1)
    )
    noobj_conf_loss = F.cross_entropy(
        pred_confidence[noobj_mask], ann_confidence[noobj_mask].argmax(dim=1)
    )
    confidence_loss = obj_conf_loss + 3 * noobj_conf_loss

    box_loss = F.smooth_l1_loss(pred_box[obj_mask], ann_box[obj_mask])

    return confidence_loss + box_loss


def conv_block(in_channels, out_channels, kernel_size, stride, padding, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()

        self.class_num = class_num  # num_of_classes, in this assignment, 4: cat, dog, person, background

        # TODO: define layers
        self.layers = nn.Sequential(
            conv_block(3, 64, kernel_size=3, stride=2, padding=1),  # 1
            conv_block(64, 64, kernel_size=3, stride=1, padding=1),  # 2
            conv_block(64, 64, kernel_size=3, stride=1, padding=1),  # 3
            conv_block(64, 128, kernel_size=3, stride=2, padding=1),  # 4
            conv_block(128, 128, kernel_size=3, stride=1, padding=1),  # 5
            conv_block(128, 128, kernel_size=3, stride=1, padding=1),  # 6
            conv_block(128, 256, kernel_size=3, stride=2, padding=1),  # 7
            conv_block(256, 256, kernel_size=3, stride=1, padding=1),  # 8
            conv_block(256, 256, kernel_size=3, stride=1, padding=1),  # 9
            conv_block(256, 512, kernel_size=3, stride=2, padding=1),  # 10
            conv_block(512, 512, kernel_size=3, stride=1, padding=1),  # 11
            conv_block(512, 512, kernel_size=3, stride=1, padding=1),  # 12
            conv_block(512, 256, kernel_size=3, stride=2, padding=1),  # 13
        )

        self.red_10 = nn.Conv2d(
            in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.blue_10 = nn.Conv2d(
            in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.down_10_a = conv_block(256, 256, kernel_size=1, stride=1, padding=0)
        self.down_10_b = conv_block(256, 256, kernel_size=3, stride=2, padding=1)

        self.red_5 = nn.Conv2d(
            in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.blue_5 = nn.Conv2d(
            in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.down_5_a = conv_block(256, 256, kernel_size=1, stride=1, padding=0)
        self.down_5_b = conv_block(256, 256, kernel_size=3, stride=1, padding=0)

        self.red_3 = nn.Conv2d(
            in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.blue_3 = nn.Conv2d(
            in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.down_3_a = conv_block(256, 256, kernel_size=1, stride=1, padding=0)
        self.down_3_b = conv_block(256, 256, kernel_size=3, stride=1, padding=0)

        self.red_1 = nn.Conv2d(
            in_channels=256, out_channels=16, kernel_size=1, stride=1, padding=0
        )
        self.blue_1 = nn.Conv2d(
            in_channels=256, out_channels=16, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # input:
        # x -- images, [batch_size, 3, 320, 320]

        # x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        batch_size = x.size(dim=0)
        x = self.layers(x)

        box_10 = self.red_10(x)
        box_10 = torch.reshape(box_10, (batch_size, 16, -1))
        conf_10 = self.blue_10(x)
        conf_10 = torch.reshape(conf_10, (batch_size, 16, -1))
        x = self.down_10_a(x)
        x = self.down_10_b(x)

        box_5 = self.red_5(x)
        box_5 = torch.reshape(box_5, (batch_size, 16, -1))
        conf_5 = self.blue_5(x)
        conf_5 = torch.reshape(conf_5, (batch_size, 16, -1))
        x = self.down_5_a(x)
        x = self.down_5_b(x)

        box_3 = self.red_3(x)
        box_3 = torch.reshape(box_3, (batch_size, 16, -1))
        conf_3 = self.blue_3(x)
        conf_3 = torch.reshape(conf_3, (batch_size, 16, -1))
        x = self.down_3_a(x)
        x = self.down_3_b(x)

        box_1 = self.red_1(x)
        box_1 = torch.reshape(box_1, (batch_size, 16, -1))
        conf_1 = self.blue_1(x)
        conf_1 = torch.reshape(conf_1, (batch_size, 16, -1))

        bboxes = torch.cat((box_10, box_5, box_3, box_1), dim=2)
        confidence = torch.cat((conf_10, conf_5, conf_3, conf_1), dim=2)

        bboxes = torch.permute(bboxes, (0, 2, 1))
        confidence = torch.permute(confidence, (0, 2, 1))

        bboxes = torch.reshape(bboxes, (batch_size, -1, 4))
        confidence = torch.reshape(confidence, (batch_size, -1, 4))
        # should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        # confidence = nn.Softmax(dim=2)(confidence)
        # print("bboxes:", bboxes.shape)
        # print("confidence:", confidence.shape)
        # sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        # confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        # bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]

        return confidence, bboxes

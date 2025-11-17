from sklearn.model_selection import train_test_split
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
import numpy as np
import os
import cv2
import albumentations as A


# generate default bounding boxes -
def default_box_generator(layers, large_scale, small_scale):
    # input:
    # layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    # large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    # small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].

    # output:
    # boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.

    # TODO:
    # create an numpy array "boxes" to store default bounding boxes
    # you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    # the first dimension means number of cells, 10*10+5*5+3*3+1*1
    # the second dimension 4 means each cell has 4 default bounding boxes.
    # their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    # where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    # for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    # the last dimension 8 means each default bounding box has 8 attributes:
    # [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max], assuming that max width and height are 1

    total_cells = sum([layer * layer for layer in layers])
    boxes = np.zeros((total_cells * 4, 8))

    idx = 0
    for i, layer_size in enumerate(layers):
        cell_size = 1 / layer_size
        ssize, lsize = small_scale[i], large_scale[i]

        # for each small unit cell
        for row in range(layer_size):
            for col in range(layer_size):

                x_center = (col + 0.5) * cell_size
                y_center = (row + 0.5) * cell_size

                sizes = [
                    (ssize, ssize),
                    (lsize, lsize),
                    (lsize * np.sqrt(2), lsize / np.sqrt(2)),
                    (lsize / np.sqrt(2), lsize * np.sqrt(2)),
                ]

                for width, height in sizes:

                    x_min = max(x_center - width / 2, 0.0)
                    y_min = max(y_center - height / 2, 0.0)

                    x_max = min(x_center + width / 2, 1.0)
                    y_max = min(y_center + height / 2, 1.0)

                    boxes[idx] = [
                        x_center,
                        y_center,
                        min(1.0, width),
                        min(1.0, height),
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                    ]

                    idx += 1

    return boxes



# this is an example implementation of IOU.
# It is different from the one used in YOLO, please pay attention.
# you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min, y_min, x_max, y_max):
    # input:
    # boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    # x_min,y_min,x_max,y_max -- another box (box_r)

    # output:
    # ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]

    inter = np.maximum(
        np.minimum(boxs_default[:, 6], x_max) - np.maximum(boxs_default[:, 4], x_min), 0
    ) * np.maximum(
        np.minimum(boxs_default[:, 7], y_max) - np.maximum(boxs_default[:, 5], y_min), 0
    )
    area_a = (boxs_default[:, 6] - boxs_default[:, 4]) * (
        boxs_default[:, 7] - boxs_default[:, 5]
    )
    area_b = (x_max - x_min) * (y_max - y_min)
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-8)


def match(
    ann_box, ann_confidence, boxs_default, threshold, cat_id, x_min, y_min, x_max, y_max
):
    # input:
    # ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    # ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    # boxs_default            -- [num_of_boxes,8], default bounding boxes
    # threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    # cat_id                  -- class id, 0-cat, 1-dog, 2-person
    # x_min,y_min,x_max,y_max -- bounding box

    # compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min, y_min, x_max, y_max)

    ious_true = ious > threshold
    # TODO:
    # update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    # if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    # this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    for idx in np.where(ious_true)[0]:  # Iterate over indices of matching default boxes
        # Get the default box center and size
        px, py, pw, ph = boxs_default[idx, :4]

        # Compute relative attributes of the ground truth box with respect to the default box
        tx = ((x_min + x_max) / 2 - px) / pw
        ty = ((y_min + y_max) / 2 - py) / ph
        tw = np.log((x_max - x_min) / pw)
        th = np.log((y_max - y_min) / ph)

        # Update ann_box with these relative attributes
        ann_box[idx] = [tx, ty, tw, th]

        # Update ann_confidence to reflect the correct class
        ann_confidence[idx] = 0  # Reset current confidence
        ann_confidence[idx, cat_id] = (
            1  # Set the confidence for the corresponding class
        )

    # TODO:
    # make sure at least one default bounding box is used
    # update ann_box and ann_confidence (do the same thing as above)
    # Step 4: Ensure at least one default box is matched (fallback to highest IOU)
    if not ious_true.any():
        ious_true = np.argmax(ious)  # Index of the default box with the highest IOU
        px, py, pw, ph = boxs_default[ious_true, :4]

        # Compute relative attributes
        tx = ((x_min + x_max) / 2 - px) / pw
        ty = ((y_min + y_max) / 2 - py) / ph
        tw = np.log((x_max - x_min) / pw)
        th = np.log((y_max - y_min) / ph)

        # Update the best matching default box
        ann_box[ious_true] = [tx, ty, tw, th]
        ann_confidence[ious_true] = 0  # Reset current confidence
        ann_confidence[ious_true, cat_id] = (
            1  # Set confidence for the corresponding class
        )


class COCO(torch.utils.data.Dataset):
    def __init__(
        self,
        imgdir,
        anndir,
        class_num,
        boxs_default,
        train=True,
        image_size=320,
        final_report=False,
    ):
        self.final_report = final_report
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num

        # overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)

        # List image files with valid extensions
        valid_extensions = ".jpg"
        self.img_names = [
            f
            for f in os.listdir(self.imgdir)
            if f.lower().endswith(valid_extensions) and not f.startswith(".")
        ]

        self.image_size = image_size
        self.train_transform = self._build_train_transform()
        self.eval_transform = self._build_eval_transform()

        # notice:
        # you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

        # 4. splitting

        if self.final_report:
            pass
        else:
            img_train, img_validation = train_test_split(
                self.img_names, test_size=0.1, random_state=42
            )
            if self.train:
                self.img_names = img_train
            else:
                self.img_names = img_validation

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num, 4], np.float32)  # bounding boxes
        ann_confidence = np.zeros([self.box_num, self.class_num], np.float32)
        # one-hot vectors
        # one-hot vectors with four classes
        # [1,0,0,0] -> cat
        # [0,1,0,0] -> dog
        # [0,0,1,0] -> person
        # [0,0,0,1] -> background

        # the default class for all cells is set to "background"
        ann_confidence[:, -1] = 1

        img_name = self.imgdir + self.img_names[index]
        ann_name = self.anndir + self.img_names[index][:-3] + "txt"

        # 1. prepare the image [3,320,320], by reading image "img_name" first.
        if not os.path.exists(img_name):
            print(f"File not found: {img_name}")
        image = cv2.imread(img_name)  # (H, W, 3)
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {img_name}")
        orig_h, orig_w = image.shape[:2]

        # 2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #    ann_name: class, xmin, ymin, w, h
        #    example: 2 334.71 38.47 173.12 316.43
        #    ann_box should be: [relative_center_x, relative_center_y, relative_width, relative_height]
        boxes = []
        labels = []
        with open(ann_name, "r") as f:
            for line in f:
                class_id, x_min, y_min, w, h = map(float, line.strip().split())
                x_max = x_min + w
                y_max = y_min + h
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id))

        boxes, labels = self._clip_boxes_to_image(boxes, labels, orig_w, orig_h)
        image, boxes, labels = self._apply_transforms(image, boxes, labels)
        labels = [int(label) for label in labels]

        h, w, _ = image.shape
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image /= 255.0

        normalized_boxes = []
        for bbox in boxes:
            x_min, y_min, x_max, y_max = bbox
            normalized_boxes.append(
                [
                    x_min / w if w else 0.0,
                    y_min / h if h else 0.0,
                    x_max / w if w else 0.0,
                    y_max / h if h else 0.0,
                ]
            )

        for bbox, class_id in zip(normalized_boxes, labels):
            x_min, y_min, x_max, y_max = bbox
            match(
                ann_box,
                ann_confidence,
                self.boxs_default,
                self.threshold,
                int(class_id),
                x_min,
                y_min,
                x_max,
                y_max,
            )

        return image, ann_box, ann_confidence

    def _build_train_transform(self):
        return A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.2,
            ),
        )

    def _build_eval_transform(self):
        return A.Compose(
            [A.Resize(self.image_size, self.image_size)],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
            ),
        )

    def _apply_transforms(self, image, boxes, labels):
        transform = self.train_transform if self.train else self.eval_transform

        if not boxes:
            boxes = []
            labels = []

        transformed = transform(
            image=image,
            bboxes=boxes,
            class_labels=labels,
        )

        return (
            transformed["image"],
            transformed.get("bboxes", []),
            transformed.get("class_labels", []),
        )

    def _clip_boxes_to_image(self, boxes, labels, width, height):
        clipped_boxes = []
        clipped_labels = []
        max_x = max(width - 1, 0)
        max_y = max(height - 1, 0)
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            x_min = min(max(x_min, 0.0), max_x)
            y_min = min(max(y_min, 0.0), max_y)
            x_max = min(max(x_max, 0.0), max_x)
            y_max = min(max(y_max, 0.0), max_y)

            if x_max < x_min:
                x_max = x_min
            if y_max < y_min:
                y_max = y_min

            if x_max == x_min or y_max == y_min:
                # Degenerate box, skip so Albumentations doesn't fail
                continue

            clipped_boxes.append([x_min, y_min, x_max, y_max])
            clipped_labels.append(label)
        return clipped_boxes, clipped_labels

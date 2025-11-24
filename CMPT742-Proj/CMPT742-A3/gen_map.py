import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from dataset import *
from model import *
from utils import *

class_num = 4  # cat dog person background

num_epochs = 64
boxs_default = default_box_generator(
    [10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Using device: {device}")

network = SSD(class_num)
network.to(device)
cudnn.benchmark = True if device.type == "cuda" else False

dataset_test = COCO(
    "data/train/images/",
    "data/train/annotations/",
    class_num,
    boxs_default,
    train=False,
    image_size=320,
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=8, shuffle=False, num_workers=0
)

network.load_state_dict(torch.load("network.pth", map_location=device))
network.eval()


def generate_mAP(dataloader, network, class_names):
    all_pred_confidences = []
    all_ann_confidences = []

    for data in dataloader:
        images_, _, ann_confidence_ = data
        images = images_.to(device)
        ann_confidence = ann_confidence_.numpy()

        pred_confidence, _ = network(images)
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()

        all_pred_confidences.append(pred_confidence_)
        all_ann_confidences.append(ann_confidence[0])

    all_pred_confidences = np.concatenate(all_pred_confidences, axis=0)
    all_ann_confidences = np.concatenate(all_ann_confidences, axis=0)

    precision_list = []
    recall_list = []
    average_precision_list = []

    for class_index in range(len(class_names) - 1):
        true_class = all_ann_confidences[:, class_index]
        if np.sum(true_class) == 0:
            continue
        pred_scores_class = all_pred_confidences[:, class_index]

        precision, recall, _ = precision_recall_curve(true_class, pred_scores_class)

        ap = average_precision_score(true_class, pred_scores_class)

        precision_list.append(precision)
        recall_list.append(recall)
        average_precision_list.append(ap)

        plt.plot(recall, precision, label=f"{class_names[class_index]} (AP = {ap:.2f})")

    mAP = np.mean(average_precision_list)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for All Classes")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"mAP-precision-recall-curve.png")
    plt.close()

    print(f"Mean Average Precision (mAP) across dataset: {mAP:.4f}")

    return mAP


mAP = generate_mAP(dataloader_test, network, ["cat", "dog", "person", "background"])

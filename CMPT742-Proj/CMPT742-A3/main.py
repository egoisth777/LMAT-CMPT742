import argparse
import os
import numpy as np
import time
import cv2

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

from dataset import *
from model import *
from utils import *
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
args = parser.parse_args()
# please google how to use argparse
# a short intro:
# to train: python main.py
# to test:  python main.py --test


class_num = 4  # cat dog person background

num_epochs = 100  # was 100
# batch_size = 128
batch_size = 128


boxs_default = default_box_generator(
    [10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Using device: {device}")

network = SSD(class_num)
network.to(device)
cudnn.benchmark = True if device.type == "cuda" else False


if not args.test:
    dataset = COCO(
        "data/train/images/",
        "data/train/annotations/",
        class_num,
        boxs_default,
        train=True,
        image_size=320,
    )
    dataset_test = COCO(
        "data/train/images/",
        "data/train/annotations/",
        class_num,
        boxs_default,
        train=False,
        image_size=320,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=0
    )

    optimizer = optim.Adam(network.parameters(), lr=1e-3 * 0.5)
    # feel free to try other optimizers and parameters.

    start_time = time.time()

    for epoch in range(num_epochs):
        # TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0

        train_loader = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True
        )
        for i, data in enumerate(train_loader, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.to(device)
            ann_box = ann_box_.to(device)
            ann_confidence = ann_confidence_.to(device)

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            # if i == 0:
            #     pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            #     np.savetxt(
            #         "debug/visualize_pred/pred_confidence2.txt",
            #         pred_confidence_,
            #         fmt="%.2f",
            #     )
            #     pred_box_ = pred_box[0].detach().cpu().numpy()
            #     np.savetxt(
            #         "debug/visualize_pred/pred_box2.txt",
            #         pred_box_,
            #         fmt="%.2f",
            #     )
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()

            avg_loss += loss_net.data
            avg_count += 1
            train_loader.set_postfix({"train loss": avg_loss / avg_count})

        print(
            "[%d] time: %f train loss: %f"
            % (epoch, time.time() - start_time, avg_loss / avg_count)
        )

        # visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        confidence_tensor = torch.tensor(pred_confidence_, dtype=torch.float32)
        confidence_softmax = F.softmax(confidence_tensor, dim=1)
        pred_confidence_ = non_maximum_suppression(
            confidence_softmax, pred_box_, boxs_default
        )

        visualize_pred(
            "train",
            pred_confidence_,
            pred_box_,
            ann_confidence_[0].numpy(),
            ann_box_[0].numpy(),
            images_[0].numpy(),
            boxs_default,
        )

        # VALIDATION
        network.eval()

        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate

        with torch.no_grad():
            val_loader = tqdm(
                dataloader_test, desc=f"Validation {epoch+1}/{num_epochs}", leave=True
            )
            val_loss = 0
            val_count = 0
            for i, data in enumerate(val_loader, 0):
                images_, ann_box_, ann_confidence_ = data
                images = images_.to(device)
                ann_box = ann_box_.to(device)
                ann_confidence = ann_confidence_.to(device)

                pred_confidence, pred_box = network(images)

                pred_confidence_ = pred_confidence.detach().cpu().numpy()
                pred_box_ = pred_box.detach().cpu().numpy()

                # optional: imp ement a function to accumulate precision and recall to compute mAP or F1.
                # update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
                loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)

                val_loss += loss_net.data
                val_count += 1
                val_loader.set_postfix({"val_loss": val_loss / val_count})

            print(
                "[%d] time: %f validation loss: %f"
                % (epoch, time.time() - start_time, val_loss / val_count)
            )

            # visualize
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()

            confidence_tensor = torch.tensor(pred_confidence_, dtype=torch.float32)
            confidence_softmax = F.softmax(confidence_tensor, dim=1)
            pred_confidence_ = non_maximum_suppression(
                confidence_softmax, pred_box_, boxs_default
            )

            visualize_pred(
                "val_",
                pred_confidence_,
                pred_box_,
                ann_confidence_[0].numpy(),
                ann_box_[0].numpy(),
                images_[0].numpy(),
                boxs_default,
            )

        # optional: compute F1
        # F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        # print(F1score)

        # save weights
        if epoch % 10 == 9:
            # save last network
            print("saving net...")
            torch.save(network.state_dict(), "network.pth")


else:
    # TEST
    dataset_test = COCO(
        "data/test/images/",
        "data/train/annotations/",
        class_num,
        boxs_default,
        train=False,
        image_size=320,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0
    )

    # network.load_state_dict(torch.load("network.pth"))
    network.load_state_dict(torch.load("network.pth", map_location=device))
    network.eval()

    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_ = data
        images = images_.to(device)
        ann_box = ann_box_.to(device)
        ann_confidence = ann_confidence_.to(device)

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        confidence_tensor = torch.tensor(pred_confidence_, dtype=torch.float32)
        confidence_softmax = F.softmax(confidence_tensor, dim=1)
        pred_confidence_ = non_maximum_suppression(
            confidence_softmax, pred_box_, boxs_default
        )
        # TODO: save predicted bounding boxes and classes to a txt file.
        # you will need to submit those files for grading this assignment
        with open("output/pred/predictions_" + str(i) + ".txt", "w") as f:
            for pc in range(len(pred_confidence_)):
                for cn in range(0, 3):
                    # to match the threshold in visualization
                    if pred_confidence_[pc, cn] > 0.9:
                        x_min, y_min, x_max, y_max = pred_box_[pc]
                        cls = cn
                        # in the format: class_id x_min y_min x_max y_max
                        f.write(
                            f"{cls} {x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f}\n"
                        )

        visualize_pred(
            "test_image_" + str(i),
            pred_confidence_,
            pred_box_,
            ann_confidence_[0].numpy(),
            ann_box_[0].numpy(),
            images_[0].numpy(),
            boxs_default,
        )
        # cv2.waitKey(1000)
        cv2.destroyAllWindows()

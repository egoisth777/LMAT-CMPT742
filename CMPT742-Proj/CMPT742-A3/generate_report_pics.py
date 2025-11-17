import cv2
import torch

from dataset import *
from model import *
from utils import *
from utils import _safe_destroy_windows

class_num = 4  # cat dog person background

boxs_default = default_box_generator(
    [10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Using device: {device}")

network = SSD(class_num)
network.to(device)


dataset_test = COCO(
    "report_test_images/",
    "data/train/annotations/",
    class_num,
    boxs_default,
    train=False,
    image_size=320,
    final_report=True,
)


dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0
)
print(f"Total test samples: {len(dataset_test)}")

network.load_state_dict(torch.load("network.pth", map_location=device))
network.eval()

for i, data in enumerate(dataloader_test, 0):
    images_, ann_box_, ann_confidence_ = data
    images = images_.to(device)
    ann_box = ann_box_.to(device)
    ann_confidence = ann_confidence_.to(device)

    pred_confidence, pred_box = network(images)

    # Move predictions to CPU for further processing
    pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
    pred_box_ = pred_box[0].detach().cpu().numpy()

    confidence_tensor = torch.tensor(pred_confidence_, dtype=torch.float32)
    confidence_softmax = F.softmax(confidence_tensor, dim=1)
    pred_confidence_ = non_maximum_suppression(
        confidence_softmax, pred_box_, boxs_default
    )
    # np.savetxt("debug/confidence_softmax.txt", confidence_softmax, fmt="%.2f")

    visualize_pred(
        "report_" + str(i),
        pred_confidence_,
        pred_box_,
        ann_confidence_[0].numpy(),
        ann_box_[0].numpy(),
        images_[0].numpy(),
        boxs_default,
        True,
    )
    # cv2.waitKey(1000)
    _safe_destroy_windows()

import numpy as np
import cv2
import torch
from dataset import iou
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# use [blue green red] to represent different classes
classes = ["cat", "dog", "person", "background"]


def decode_box(ann_box, boxs_default):
    """
    Decode ann_box (encoded as [tx, ty, tw, th]) back to image coordinates.
    """
    decoded_boxes = []
    for i, (tx, ty, tw, th) in enumerate(ann_box):
        px, py, pw, ph = boxs_default[i, :4]  # Default box prameters
        # print("px, py, pw, ph", px, py, pw, ph)
        # px, py, pw, ph 0.5 0.5 1.0 0.565685424949238
        # Decode center, width, and height
        cx = tx * pw + px
        cy = ty * ph + py
        w = np.exp(tw) * pw
        h = np.exp(th) * ph
        # print("wh", w,h)
        # wh 0.565685424949238 1.0

        # Convert to corner coordinates
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
        # print("x_min", x_min, y_min, x_max, y_max)
        # x_min 0.217157287525381 0.0 0.782842712474619 1.0

        decoded_boxes.append([x_min, y_min, x_max, y_max])
    return np.array(decoded_boxes)


def visualize_pred(
    windowname,
    pred_confidence,
    pred_box,
    ann_confidence,
    ann_box,
    image_,
    boxs_default,
    final_report=False,
):

    # np.savetxt(
    #     "debug/visualize_pred/pred_confidence_input" + windowname + ".txt",
    #     pred_confidence,
    #     fmt="%.2f",
    # )
    # print(pred_confidence.shape)

    image_ = np.clip(image_ * 255, 0, 255).astype(np.uint8)

    # input:
    # windowname      -- the name of the window to display the images

    # pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]

    # ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]

    # boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    # image_          -- the input image to the network 3 320 320

    # np.savetxt("debug/visualize_pred/pred_confidence.txt", pred_confidence, fmt="%.2f")
    # np.savetxt("debug/visualize_pred/pred_box.txt", pred_box, fmt="%.2f")
    # np.savetxt("debug/visualize_pred/ann_confidence.txt", ann_confidence, fmt="%.2f")
    # np.savetxt("debug/visualize_pred/ann_box.txt", ann_box, fmt="%.2f")
    # np.savetxt("debug/visualize_pred/boxs_default.txt", boxs_default, fmt="%.2f")

    _, class_num = pred_confidence.shape
    # class_num = 4
    class_num = class_num - 1
    # class_num = 3 now, because we do not need the last class (background)

    image = np.transpose(image_, (1, 2, 0)).astype(np.uint8)
    image1 = np.zeros(image.shape, np.uint8)
    image2 = np.zeros(image.shape, np.uint8)
    image3 = np.zeros(image.shape, np.uint8)
    image4 = np.zeros(image.shape, np.uint8)
    image1[:] = image[:]
    image2[:] = image[:]
    image3[:] = image[:]
    image4[:] = image[:]
    # image1: draw ground truth bounding boxes on image1
    # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    # image3: draw network-predicted bounding boxes on image3
    # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    h, w, _ = image1.shape
    decoded_ann_box = decode_box(ann_box, boxs_default)
    decoded_pred_box = decode_box(pred_box, boxs_default)

    font_scale = 0.8
    thickness = 2

    # draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if (
                ann_confidence[i, j] > 0.5
            ):  # if the network/ground_truth has high confidence on cell[i] with class[j]
                # TODO:
                # image1: draw ground truth bounding boxes on image1
                # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)

                # you can use cv2.rectangle as follows:
                # start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                # end_point = (x2, y2) #bottom right corner
                # color = colors[j] #use red green blue to represent different classes
                # thickness = 2
                # cv2.rectangle(image?, start_point, end_point, color, thickness)
                # Draw ground truth bounding boxes on image1
                x1, y1, x2, y2 = decoded_ann_box[i]
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                start_point = (x1, y1)
                end_point = (x2, y2)
                cv2.rectangle(image1, start_point, end_point, colors[j], 2)

                # Draw default boxes on image2
                x1, y1, x2, y2 = boxs_default[i][-4:]
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                start_point = (x1, y1)
                end_point = (x2, y2)
                cv2.rectangle(image2, start_point, end_point, colors[j], 2)

    # pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i, j] > 0.9:
                # TODO:
                # image3: draw network-predicted bounding boxes on image3
                # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                x1, y1, x2, y2 = decoded_pred_box[i]
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                start_point = (x1, y1)
                end_point = (x2, y2)
                cv2.rectangle(image3, start_point, end_point, colors[j], 2)

                text_x = x1  # Align with the left corner of the rectangle
                text_y = y2 + 20

                if text_y > image3.shape[1]:
                    text_y = y2 - 20  # Adjust below the rectangle if out of bounds

                # Put the text on the image
                cv2.putText(
                    image3,
                    f"{classes[j]}: {pred_confidence[i, j].item():.2f}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colors[j],
                    thickness,
                )

                # Draw predicted default boxes on image4
                x1, y1, x2, y2 = boxs_default[i][:4]
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                start_point = (x1, y1)
                end_point = (x2, y2)
                cv2.rectangle(image4, start_point, end_point, colors[j], 2)

    # combine four images into one

    image = np.zeros([h * 2, w * 2, 3], np.uint8)

    # Add titles to the individual images
    cv2.putText(
        image1,
        "Ground Truth Boxes",
        (10, 20),  # Adjust starting position for smaller text
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colors[2],
        thickness,
    )
    cv2.putText(
        image2,
        "Ground Truth Default Boxes",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colors[2],
        thickness,
    )
    cv2.putText(
        image3,
        "Predicted Boxes",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colors[2],
        thickness,
    )
    cv2.putText(
        image4,
        "Predicted Default Boxes",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colors[2],
        thickness,
    )

    image[:h, :w] = image1
    image[:h, w:] = image2
    image[h:, :w] = image3
    image[h:, w:] = image4
    # cv2.imshow(windowname + " [[gt_box,gt_dft],[pd_box,pd_dft]]", image)
    if final_report:
        cv2.imwrite("report_output/" + windowname + "_visualized.jpg", image)

    else:
        cv2.imwrite("output/images/" + windowname + "_visualized.jpg", image)
    # cv2.waitKey(0)
    _safe_destroy_windows()
    # if you are using a server, you may not be able to display the image.
    # in that case, please save the image using cv2.imwrite and check the saved image for visualization.


def non_maximum_suppression(
    confidence_, box_, boxs_default, overlap=0.2, threshold=0.8
):

    if isinstance(confidence_, torch.Tensor):
        confidence = confidence_.detach().cpu().numpy()
    else:
        confidence = np.array(confidence_, copy=True)

    decoded_boxes = decode_box(box_, boxs_default)
    num_classes = confidence.shape[1] - 1  # ignore background
    suppressed_pred = np.zeros_like(confidence)

    def _compute_iou(base_box, other_boxes):
        if other_boxes.size == 0:
            return np.array([])
        x_min = np.maximum(base_box[0], other_boxes[:, 0])
        y_min = np.maximum(base_box[1], other_boxes[:, 1])
        x_max = np.minimum(base_box[2], other_boxes[:, 2])
        y_max = np.minimum(base_box[3], other_boxes[:, 3])
        inter_w = np.clip(x_max - x_min, a_min=0.0, a_max=None)
        inter_h = np.clip(y_max - y_min, a_min=0.0, a_max=None)
        inter_area = inter_w * inter_h

        base_area = (base_box[2] - base_box[0]) * (base_box[3] - base_box[1])
        other_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (
            other_boxes[:, 3] - other_boxes[:, 1]
        )
        union = base_area + other_area - inter_area
        return inter_area / np.clip(union, a_min=1e-8, a_max=None)

    for cls in range(num_classes):
        scores = confidence[:, cls]
        candidate_idxs = np.where(scores >= threshold)[0]
        if candidate_idxs.size == 0:
            continue

        order = candidate_idxs[np.argsort(scores[candidate_idxs])[::-1]]
        keep = []

        while order.size > 0:
            current = order[0]
            keep.append(current)
            order = order[1:]

            if order.size == 0:
                break

            ious = _compute_iou(decoded_boxes[current], decoded_boxes[order])
            order = order[ious <= overlap]

        suppressed_pred[keep, cls] = scores[keep]

    background_mask = np.sum(suppressed_pred[:, :-1], axis=1) == 0
    suppressed_pred[background_mask, -1] = 1.0

    return suppressed_pred


def _safe_destroy_windows():
    """Best-effort cleanup that won't crash when HighGUI isn't available."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


###
# for the `generate_mAP` method, it's in `map.py`. run `python map.py` will do the job.
###
# def generate_mAP(dataloader, network, class_names):
# it's in `map.py`.

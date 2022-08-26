import torch
from collections import Counter
from iou import intersection_over_union

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5,
                           box_formats="corners", num_classes=20):
    #pred_boxes = [[train_idx, class_pred, prob_score, x1, y1, x2, y2],...]
    average_precision = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)


        amount_boxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_boxes.items():
            amount_boxes[key] = torch.zeros(val)


        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):




# TODO: Implement IoU with handling for closest guess (Hungarian Algorithm?)
# Implement mAP, precision, recall, F1 score
# Hungarian Algorithm asses the overlap of all predicted boxes with true boxes and class labels 
# to assign map predections with truth and assess False Positives and False Negatives.
import torch
from scipy.optimize import linear_sum_assignment
import torchvision.ops as ops
# https://blog.roboflow.com/object-detection-metrics/
# https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html


def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Matches GT and Pred boxes under missing/extra constraints using the Hungarian Algorithm.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    # Step 1: Calculate the N x M IoU Matrix
    iou_matrix = ops.box_iou(gt_boxes, pred_boxes).numpy()
    
    # Step 2: Convert to cost matrix for minimization (1 - IoU)
    cost_matrix = 1.0 - iou_matrix
    
    # Step 3: Find optimal 1-to-1 matching indices
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    
    matched_pairs = []
    unmatched_gt = set(range(len(gt_boxes)))
    unmatched_pred = set(range(len(pred_boxes)))
    
    # Step 4: Filter assignments by IoU threshold
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        iou = iou_matrix[gt_idx, pred_idx]
        if iou >= iou_threshold:
            matched_pairs.append({'gt_idx': gt_idx, 'pred_idx': pred_idx, 'iou': iou})
            unmatched_gt.remove(gt_idx)
            unmatched_pred.remove(pred_idx)
            
    return matched_pairs, list(unmatched_gt), list(unmatched_pred)


def hung_alg_test():
    # --- EXAMPLE USAGE ---
    # 3 Ground Truth Boxes
    gt = torch.tensor([
        [10, 10, 50, 50],
        [60, 60, 100, 100],
        [120, 120, 150, 150]  # This one will be MISSING (no match)
    ], dtype=torch.float32)

    # 4 Predicted Boxes (1 extra/false positive)
    pred = torch.tensor([
        [12, 12, 48, 48],    # Matches GT 0
        [58, 62, 102, 98],   # Matches GT 1
        [200, 200, 250, 250], # EXTRA box (No GT close by)
        [11, 11, 49, 49]     # EXTRA prediction overlapping GT 0 (Filtered out by Hungarian)
    ], dtype=torch.float32)


    matches, missing_gt, extra_pred = match_boxes(gt, pred, iou_threshold=0.5)

    print("✅ MATCHED PAIRS:")
    for m in matches:
        print(f"   GT Box {m['gt_idx']} <--> Pred Box {m['pred_idx']} (IoU: {m['iou']:.2f})")

    print(f"\n❌ MISSING BOXES (False Negatives): {missing_gt}")
    print(f"⚠️ EXTRA BOXES (False Positives): {extra_pred}")


# TODO: Figure out how to handle the weighting. Per image? Per object? 
# I *think* per object, so batch images and flatten results.
# Yes, this is based on object occurence. Individual images needed for separatring true postitives
# from false positives / negatives.
# TODO: This should also be calcualted per class and overall.
def calculate_precision(true_pos, false_pos):
    return true_pos / (true_pos +  false_pos) 


def calculate_recall(true_pos, false_neg):
    return true_pos / (true_pos +  false_neg) 


def calculate_mean_average_precision():
    # Use torchmetrics.detection implementation
    pass


def calculate_f1_score(precision, recall):
    return 2*(precision*recall)/(precision+recall)

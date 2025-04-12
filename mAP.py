import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.validation import explain_validity
from matplotlib.patches import Polygon as MatplotlibPolygon

def oriented_iou(box1, box2, plot=False) -> float:
    """
    Calculate the Intersection over Union (IoU) of two rectangles (possibly rotated).
    :param rect1: numpy array of shape (4, 2) representing the first rectangle (each row is [x, y]).
    :param rect2: numpy array of shape (4, 2) representing the second rectangle.
    :param plot: If True, displays a plot of the two rectangles and their intersection.
    :return: The Intersection over Union (IoU) as a float. Returns 0 if there is no union area.
    """
    rect1 = np.array(box1).reshape(4, 2)
    rect2 = np.array(box2).reshape(4, 2)
    def validate_polygon(coords):
        """Validates and fixes a polygon if necessary."""
        polygon = Polygon(coords)
        if not polygon.is_valid:
            print(f"Invalid polygon: {explain_validity(polygon)}")
            # Attempt to fix self-intersection by using the convex hull
            polygon = polygon.convex_hull
            print(f"Polygon fixed using convex hull: {polygon.wkt}")
        # Ensure the polygon is oriented counterclockwise
        return orient(polygon, sign=1.0)
    # Validate and fix input polygons
    poly1 = validate_polygon(rect1)
    poly2 = validate_polygon(rect2)
    # Compute the intersection of the two polygons
    intersection = poly1.intersection(poly2)
    # Plotting the rectangles and their intersection if requested
    if plot:
        fig, ax = plt.subplots(figsize=(3, 3))
        # Create patches for both rectangles
        rect_patch1 = MatplotlibPolygon(np.array(poly1.exterior.coords), closed=True,
                                        edgecolor='blue', facecolor='blue', alpha=0.4, label="Rectangle 1")
        rect_patch2 = MatplotlibPolygon(np.array(poly2.exterior.coords), closed=True,
                                        edgecolor='green', facecolor='green', alpha=0.4, label="Rectangle 2")
        ax.add_patch(rect_patch1)
        ax.add_patch(rect_patch2)
        # If an intersection exists, draw it
        if not intersection.is_empty:
            if intersection.geom_type == 'Polygon':
                x, y = intersection.exterior.xy
                ax.fill(x, y, color='red', alpha=0.5, label="Intersection")
            elif intersection.geom_type == 'MultiPolygon':
                for geom in intersection.geoms:
                    x, y = geom.exterior.xy
                    ax.fill(x, y, color='red', alpha=0.5, label="Intersection")
        
        # Adjust plot limits based on input rectangles
        all_points = np.vstack((rect1, rect2))
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.grid(True)
        plt.show()
    # Compute the areas needed for the IoU.
    intersection_area_val = intersection.area if not intersection.is_empty else 0.0
    union_area_val = poly1.area + poly2.area - intersection_area_val
    # Avoid division by zero; if union area is 0, return 0.
    iou = intersection_area_val / union_area_val if union_area_val > 0 else 0.0
    return iou


def axis_aligned_iou(boxA, boxB):
    """
    Compute IoU for axis-aligned boxes.
    box = [xmin, ymin, xmax, ymax].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def universal_iou(box1, box2):
    """
    Detects box type (axis-aligned or oriented) by length.
    If length == 4 => axis-aligned
    If length == 8 => oriented bounding box
    """
    # Convert to plain Python list (in case they're NumPy arrays)
    box1 = box1.tolist() if hasattr(box1, 'tolist') else box1
    box2 = box2.tolist() if hasattr(box2, 'tolist') else box2

    if len(box1) == 4 and len(box2) == 4:
        return axis_aligned_iou(box1, box2)
    elif len(box1) == 8 and len(box2) == 8:
        return oriented_iou(box1, box2)
    else:
        raise ValueError(f"Box shapes do not match known formats: "
                         f"len(box1)={len(box1)}, len(box2)={len(box2)}.")


def compute_mAP(preds, targets, box_name="boxes", iou_threshold=0.5):
    """
    Computes mean Average Precision (mAP) for object detection at a single IoU threshold.
    
    This is the ORIGINAL function requested to remain intact.
    It returns mAP across all classes at the specified IoU threshold.
    """
    # 1) Convert PyTorch tensors -> NumPy arrays
    #    Build ground-truth (gt_dict) and predictions (pred_dict) structures keyed by class.
    gt_dict = {}
    pred_dict = {}

    for img_id, (pred, target) in enumerate(zip(preds, targets)):
        # Convert to NumPy
        pred_boxes  = pred[box_name].detach().cpu().numpy()
        pred_scores = pred["scores"].detach().cpu().numpy()
        pred_labels = pred["labels"].detach().cpu().numpy()

        gt_boxes  = target[box_name].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()

        # Populate gt_dict
        for box, cls in zip(gt_boxes, gt_labels):
            gt_dict.setdefault(cls, []).append({
                "img_id": img_id,
                "box": box,
                "used": False  # to mark if this GT box is already assigned
            })

        # Populate pred_dict
        for box, score, cls in zip(pred_boxes, pred_scores, pred_labels):
            pred_dict.setdefault(cls, []).append({
                "img_id": img_id,
                "box": box,
                "score": score
            })

    # 2) Sort predictions by descending score within each class
    for cls in pred_dict:
        pred_dict[cls].sort(key=lambda x: x["score"], reverse=True)

    # 3) For each class, compute Average Precision (AP)
    average_precisions = []

    classes_in_gt = sorted(gt_dict.keys())  # classes that appear in GT
    for cls in classes_in_gt:
        # If this class has no predictions at all, AP = 0
        if cls not in pred_dict:
            average_precisions.append(0.0)
            continue

        predictions = pred_dict[cls]
        n_pred = len(predictions)

        gt_cls = gt_dict[cls]
        n_gt = len(gt_cls)

        TP = np.zeros(n_pred)
        FP = np.zeros(n_pred)

        for i, pred_item in enumerate(predictions):
            pred_box = pred_item["box"]
            pred_img_id = pred_item["img_id"]

            # Filter ground-truth boxes for same image
            candidate_gts = [g for g in gt_cls if g["img_id"] == pred_img_id]

            best_iou = 0.0
            best_gt_idx = -1
            for idx, g in enumerate(candidate_gts):
                if not g["used"]:
                    iou_val = universal_iou(pred_box, g["box"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # Mark GT box as used
                candidate_gts[best_gt_idx]["used"] = True
                TP[i] = 1
            else:
                FP[i] = 1

        # 4) Compute precision and recall
        cum_TP = np.cumsum(TP)
        cum_FP = np.cumsum(FP)

        recalls = cum_TP / float(n_gt)
        precisions = cum_TP / np.maximum((cum_TP + cum_FP), np.finfo(np.float64).eps)

        # 5) Compute AP (VOC method)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for i_ in range(mpre.size - 1, 0, -1):
            mpre[i_ - 1] = max(mpre[i_ - 1], mpre[i_])

        # Area under PR curve
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        average_precisions.append(ap)

    if len(average_precisions) == 0:
        return 0.0
    mAP = np.mean(average_precisions)
    return mAP


def compute_detection_metrics(preds, targets, box_name="boxes"):
    """
    Extended function that:
      1) Returns the old single-threshold (0.5) mAP
      2) Returns mAP50-95 (averaged over thresholds 0.5 to 0.95)
      3) Returns per-class AP at IoU=0.5
      4) Computes overall (dataset-level) precision, recall, and a custom "accuracy" at IoU=0.5

    Returns a dictionary with:
      {
        "mAP_0.5": ...,
        "mAP_50_95": ...,
        "AP_per_class_0.5": {class_id: AP, ...},
        "precision_0.5": ...,
        "recall_0.5": ...,
        "accuracy_0.5": ...
      }
    """

    # ------------------------------
    # 1) Single-threshold (0.5) mAP
    # ------------------------------
    mAP_50 = compute_mAP(preds, targets, iou_threshold=0.5)

    # -------------------------
    # 2) Compute mAP50-95
    # -------------------------
    iou_vals = np.arange(0.50, 0.96, 0.05)
    # For each IoU threshold, compute the mAP
    maps = [compute_mAP(preds, targets,box_name=box_name, iou_threshold=iou) for iou in iou_vals]
    mAP_50_95 = float(np.mean(maps))  # average across all thresholds

    # -----------------------------------
    # 3) Per-class AP at IoU=0.5
    #    We'll adapt the single-threshold code, but store class-wise AP.
    # -----------------------------------
    # We need to replicate the logic of compute_mAP but keep per-class results.
    # Build GT and predictions dicts again, but for safe re-use.
    gt_dict = {}
    pred_dict = {}

    for img_id, (pred, target) in enumerate(zip(preds, targets)):
        # Convert to NumPy
        pred_boxes  = pred[box_name].detach().cpu().numpy()
        pred_scores = pred["scores"].detach().cpu().numpy()
        pred_labels = pred["labels"].detach().cpu().numpy()

        gt_boxes  = target[box_name].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()

        # Populate gt_dict
        for box, cls in zip(gt_boxes, gt_labels):
            gt_dict.setdefault(cls, []).append({
                "img_id": img_id,
                "box": box,
                "used": False
            })

        # Populate pred_dict
        for box, score, cls in zip(pred_boxes, pred_scores, pred_labels):
            pred_dict.setdefault(cls, []).append({
                "img_id": img_id,
                "box": box,
                "score": score
            })

    # Sort predictions within each class by score desc
    for cls in pred_dict:
        pred_dict[cls].sort(key=lambda x: x["score"], reverse=True)

    classes_in_gt = sorted(gt_dict.keys())
    ap_per_class = {}

    for cls in classes_in_gt:
        # clone ground-truth so that "used" flags for one class won't mess up another
        # (Though typically we separate by class anyway, but let's be safe.)
        gt_cls = gt_dict[cls]
        # If no predictions for this cls, AP=0
        if cls not in pred_dict:
            ap_per_class[cls] = 0.0
            continue

        predictions = pred_dict[cls]
        n_pred = len(predictions)
        n_gt   = len(gt_cls)

        TP = np.zeros(n_pred)
        FP = np.zeros(n_pred)

        # We'll make a fresh copy of the 'used' flags so that we don't
        # permanently modify them.
        local_gt = []
        for g in gt_cls:
            local_gt.append({"img_id": g["img_id"], "box": g["box"], "used": False})

        for i, pred_item in enumerate(predictions):
            pred_box = pred_item["box"]
            pred_img_id = pred_item["img_id"]

            # filter ground-truth boxes for same image
            candidate_gts = [g for g in local_gt if g["img_id"] == pred_img_id]

            best_iou = 0.0
            best_gt_idx = -1
            for idx, g in enumerate(candidate_gts):
                if not g["used"]:
                    iou_val = universal_iou(pred_box, g["box"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = idx

            if best_iou >= 0.5 and best_gt_idx >= 0:
                candidate_gts[best_gt_idx]["used"] = True
                TP[i] = 1
            else:
                FP[i] = 1

        cum_TP = np.cumsum(TP)
        cum_FP = np.cumsum(FP)
        recalls = cum_TP / float(n_gt)
        precisions = cum_TP / np.maximum((cum_TP + cum_FP), np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for i_ in range(mpre.size - 1, 0, -1):
            mpre[i_ - 1] = max(mpre[i_ - 1], mpre[i_])

        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        ap_per_class[cls] = ap

    # ----------------------------------------
    # 4) Overall precision, recall, "accuracy"
    #    at IoU=0.5 (dataset-level).
    # ----------------------------------------
    # We combine TPs/FPs across *all* classes and images at once.
    # Re-run a single pass at IoU=0.5, but track global TP, FP, GT_count.
    total_gt = 0
    all_predictions = []

    # Build a fresh GT/pred dictionary so we can do a single pass:
    gt_dict_global = []
    for img_id, (pred, target) in enumerate(zip(preds, targets)):
        pred_boxes  = pred[box_name].detach().cpu().numpy()
        pred_scores = pred["scores"].detach().cpu().numpy()
        pred_labels = pred["labels"].detach().cpu().numpy()

        gt_boxes  = target[box_name].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()

        for box, cls in zip(gt_boxes, gt_labels):
            gt_dict_global.append({
                "img_id": img_id,
                "box": box,
                "label": cls,
                "used": False
            })
        for box, score, cls in zip(pred_boxes, pred_scores, pred_labels):
            all_predictions.append({
                "img_id": img_id,
                "box": box,
                "score": score,
                "label": cls
            })

    total_gt = len(gt_dict_global)
    # Sort all predictions by descending score
    all_predictions.sort(key=lambda x: x["score"], reverse=True)


    TP_global = 0
    FP_global = 0

    for pred_item in all_predictions:
        pred_box = pred_item["box"]
        pred_img_id = pred_item["img_id"]

        # Among all ground truths, find best match in same image and same class:
        candidate_gts = [
            g for g in gt_dict_global 
            if (g["img_id"] == pred_img_id 
                and g["label"] == pred_item["label"]
                and not g["used"])
        ]
        best_iou = 0.0
        best_gt_idx = -1
        for idx, g in enumerate(candidate_gts):
            iou_val = universal_iou(pred_box, g["box"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = idx

        if best_iou >= 0.5 and best_gt_idx >= 0:
            # Directly mark the matched GT as used
            candidate_gts[best_gt_idx]["used"] = True
            TP_global += 1
        else:
            FP_global += 1

    # precision = TP / (TP + FP)
    # recall    = TP / total_gt
    # "accuracy" = TP / (TP + FP + FN);  FN = total_gt - TP
    precision_05 = TP_global / max((TP_global + FP_global), 1e-6)
    recall_05    = TP_global / max(total_gt, 1e-6)
    fn           = total_gt - TP_global
    accuracy_05  = TP_global / max(TP_global + FP_global + fn, 1e-6)

    # -----------------------------------
    # Final result dictionary
    # -----------------------------------
    return {
        "mAP_0.5": float(mAP_50),         # single IoU=0.5
        "mAP_50_95": float(mAP_50_95),    # average across IoU=0.50:0.05:0.95
        "AP_per_class_0.5": ap_per_class, # e.g. {class_id: AP, ...}
        "precision_0.5": float(precision_05),
        "recall_0.5": float(recall_05),
        "accuracy_0.5": float(accuracy_05)
    }


'''

preds = [{
        'boxes': torch.tensor([[2.3584e-01, 1.1876e+02, 1.0015e+02, 3.1960e+02],
                                [2.4955e+02, 8.2115e+01, 4.1607e+02, 2.5002e+02],
                                [1.3821e+01, 2.4666e+02, 2.0969e+02, 4.5468e+02]]),
        'oboxes': torch.tensor([[346., 412., 391., 457., 358., 490., 313., 445.],
                                 [288., 261., 236., 122., 400., 61., 452., 198.],
                                 [5., 250., -7., 169., 201., 135., 214., 216.],
                                 [188., 456., 81., 455., 84., 237., 191., 238.],
                                 [-85., 120., 100., 120., 100., 322., -85., 322.],
                                 [15., 254., 208., 254., 208., 456., 15., 456.],
                                 [250., 652., 192., 459., 584., 342., 642., 534.]]),
        'scores': torch.tensor([0.6221, 0.5505, 0.2892]),
        'labels': torch.tensor([23, 4, 0])},
    {
        'boxes': torch.tensor([[183.9808, 254.6527, 423.7566, 422.0714],
                                [528.4429, 114.6335, 637.8246, 252.1907],
                                [395.4819, 337.3660, 624.4421, 478.7167],
                                [277.6788, 81.2580, 527.4275, 212.1042],
                                [164.0251, 378.5693, 339.7728, 544.9041]]),
        'oboxes': torch.tensor([[346., 412., 391., 457., 358., 490., 313., 445.],
                                 [288., 261., 236., 122., 400., 61., 452., 198.],
                                 [5., 250., -7., 169., 201., 135., 214., 216.],
                                 [188., 456., 81., 455., 84., 237., 191., 238.],
                                 [-85., 120., 100., 120., 100., 322., -85., 322.],
                                 [15., 254., 208., 254., 208., 456., 15., 456.],
                                 [250., 652., 192., 459., 584., 342., 642., 534.]]),
        'scores': torch.tensor([0.9047, 0.8040, 0.4532, 0.4329, 0.3631]),
        'labels': torch.tensor([22, 11, 71, 17, 74])},
    {
        'boxes': torch.tensor([[32.3895, 114.9058, 341.3117, 485.5364],
                                [71.1628, 343.4698, 275.4551, 515.8247]]),
        'oboxes': torch.tensor([[346., 412., 391., 457., 358., 490., 313., 445.],
                                 [288., 261., 236., 122., 400., 61., 452., 198.],
                                 [5., 250., -7., 169., 201., 135., 214., 216.],
                                 [188., 456., 81., 455., 84., 237., 191., 238.],
                                 [-85., 120., 100., 120., 100., 322., -85., 322.],
                                 [15., 254., 208., 254., 208., 456., 15., 456.],
                                 [250., 652., 192., 459., 584., 342., 642., 534.]]),
        'scores': torch.tensor([0.8606, 0.6822]),
        'labels': torch.tensor([23, 62])}]

targets = [{
        'boxes': torch.tensor([[332., 416., 381., 485.],
                                [250., 82., 413., 246.],
                                [-3., 144., 209., 239.],
                                [83., 238., 191., 457.],
                                [-85., 120., 101., 322.],
                                [15., 254., 208., 456.],
                                [224., 396., 621., 608.]]),
        'oboxes': torch.tensor([[346., 412., 391., 457., 358., 490., 313., 445.],
                                 [288., 261., 236., 122., 400., 61., 452., 198.],
                                 [5., 250., -7., 169., 201., 135., 214., 216.],
                                 [188., 456., 81., 455., 84., 237., 191., 238.],
                                 [-85., 120., 100., 120., 100., 322., -85., 322.],
                                 [15., 254., 208., 254., 208., 456., 15., 456.],
                                 [250., 652., 192., 459., 584., 342., 642., 534.]]),
        'labels': torch.tensor([16, 4, 66, 10, 23, 77, 14])},
    {
        'boxes': torch.tensor([[101., 356., 224., 420.],
                                [278., 49., 525., 213.],
                                [-147., 411., 402., 630.],
                                [298., 243., 574., 324.],
                                [153., 312., 396., 547.],
                                [526., 115., 634., 249.],
                                [184., 254., 422., 419.]]),
        'oboxes': torch.tensor([[105., 426., 98., 365., 220., 351., 226., 412.],
                                 [278., 49., 525., 49., 525., 212., 278., 212.],
                                 [-133., 661., -162., 453., 391., 376., 419., 584.],
                                 [303., 338., 295., 258., 572., 230., 580., 311.],
                                 [153., 312., 397., 312., 397., 548., 153., 548.],
                                 [658., 232., 552., 268., 507., 135., 614., 99.],
                                 [184., 254., 423., 254., 423., 420., 184., 420.]]),
        'labels': torch.tensor([71, 17, 14, 8, 74, 11, 22])},
    {
        'boxes': torch.tensor([[296., 206., 302., 215.],
                                [431., 368., 659., 552.],
                                [330., 30., 532., 446.],
                                [34., 112., 340., 488.],
                                [288., 208., 310., 248.],
                                [256., 153., 302., 190.],
                                [71., 347., 277., 515.]]),
        'oboxes': torch.tensor([[300., 214., 295., 213., 297., 205., 302., 206.],
                                 [431., 368., 659., 368., 659., 553., 431., 553.],
                                 [330., 30., 532., 30., 532., 445., 330., 445.],
                                 [34., 112., 341., 112., 341., 488., 34., 488.],
                                 [288., 208., 310., 208., 310., 248., 288., 248.],
                                 [256., 153., 303., 153., 303., 191., 256., 191.],
                                 [71., 347., 278., 347., 278., 516., 71., 516.]]),
        'labels': torch.tensor([14, 74, 75, 23, 19, 74, 62])}]





bbox_name = "boxes" #the key for the bounding "oboxes".
single_map_95 = compute_mAP(preds, targets, box_name=bbox_name, iou_threshold=0.95)
print(f"Single-threshold mAP@0.95: {single_map_95:.4f}")
single_map_75 = compute_mAP(preds, targets, iou_threshold=0.75)
print(f"Single-threshold mAP@0.75: {single_map_75:.4f}")
# The extended metrics (includes mAP@0.5, mAP50-95, per-class AP, precision, recall, accuracy):
all_metrics = compute_detection_metrics(preds, targets, box_name=bbox_name)
print("Extended detection metrics:")
for k, v in all_metrics.items():
    if isinstance(v, dict):
        print(f"  {k} = {{")
        for c in sorted(v.keys()):
            print(f"    class {c}: {v[c]:.4f}")
        print("  }")
    else:
        print(f"  {k}: {v:.4f}")

'''

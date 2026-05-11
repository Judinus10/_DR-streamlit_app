import numpy as np
from PIL import Image

import utils_classification as cls_utils
import utils_segmentation as seg_utils


def lesion_area_score(seg_result):
    ex_area = np.mean(seg_result["ex_mask"] > 0)
    he_area = np.mean(seg_result["he_mask"] > 0)

    return {
        "ex_area": float(ex_area),
        "he_area": float(he_area),
        "total_lesion_area": float(ex_area + he_area),
    }


def lesion_aware_predict(
    pil_img: Image.Image,
    cls_model,
    cls_device,
    seg_model,
    seg_device,
):
    # 1. Classification prediction
    cls_result = cls_utils.predict(cls_model, cls_device, pil_img)

    # 2. Segmentation prediction
    seg_result = seg_utils.predict_segmentation(seg_model, seg_device, pil_img)

    # 3. Lesion area calculation
    lesion_score = lesion_area_score(seg_result)

    pred_idx = int(cls_result["pred_idx"])
    probs = cls_result["probs"].copy()

    # 4. Simple rule-based lesion support
    total_lesion = lesion_score["total_lesion_area"]
    he_area = lesion_score["he_area"]

    adjusted_pred_idx = pred_idx
    reason = "Classification result kept."

    if total_lesion > 0.03 and pred_idx == 0:
        adjusted_pred_idx = 1
        reason = "Lesion masks detected visible lesions, so prediction was adjusted from No DR to Mild."

    if he_area > 0.05 and pred_idx < 2:
        adjusted_pred_idx = 2
        reason = "Haemorrhage area was high, so prediction was adjusted toward Moderate DR."

    return {
        **cls_result,
        "original_pred_idx": pred_idx,
        "original_pred_name": cls_utils.CLASS_NAMES[pred_idx],
        "adjusted_pred_idx": adjusted_pred_idx,
        "adjusted_pred_name": cls_utils.CLASS_NAMES[adjusted_pred_idx],
        "lesion_score": lesion_score,
        "seg": seg_result,
        "lesion_reason": reason,
    }
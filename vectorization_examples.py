import numpy as np


def calculate_areas(data: np.ndarray) -> np.ndarray:
    areas = (data[:, 2] - data[:, 0]) * (data[:, 3] - data[:, 1])

    areas = np.clip(areas, a_min=0, a_max=None)

    return areas


def best_match_per_row(boxes_a: np.ndarray, boxes_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each box in `boxes_a`, find the box in `boxes_b` with the highest IoU.

    Args
    ----
    boxes_a : np.ndarray, shape [N, 4], (x1, y1, x2, y2)
    boxes_b : np.ndarray, shape [M, 4], (x1, y1, x2, y2)

    Returns
    -------
    best_idx : np.ndarray, shape [N], int   — index in `boxes_b` of the best match for each row of `boxes_a`
    best_iou : np.ndarray, shape [N], float — corresponding IoU value, in [0, 1]
    """
    # TODO: broadcast — no python-level loops over boxes

    ba = boxes_a[:, None, :]
    bb = boxes_b[None, :, :]

    lb = np.maximum(ba[:, :, :2], bb[:, :, :2])
    rt = np.minimum(ba[:, :, 2:], bb[:, :, 2:])

    valid = np.clip(rt - lb, a_min=0, a_max=None)
    inter_areas = valid[:, :, 0] * valid[:, :, 1]

    areas_a = calculate_areas(boxes_a)
    areas_b = calculate_areas(boxes_b)

    iou = inter_areas / (areas_a[:, None] + areas_b[None, :] - inter_areas + 1e-10)

    return np.argmax(iou, axis=1), np.max(iou, axis=1)




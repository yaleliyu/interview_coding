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


import numpy as np


def masked_mean_xy(trajectories, validity):
    """
    Args
    ----
    trajectories : np.ndarray, shape [B, A, T, 2]
    validity     : np.ndarray (bool), shape [B, A, T]

    Returns
    -------
    np.ndarray, shape [B, A, 2] — mean (x, y) over valid timesteps,
    or (0, 0) if no valid timesteps exist for that agent.
    """
    # TODO: mask invalid positions to 0, sum, divide by count, guard against zero-count

    masked = np.where(validity[..., None], trajectories, 0.0)        # [B,A,T,2]
    summed = masked.sum(axis=-2)                                      # [B,A,2]
    count = validity.sum(axis=-1, keepdims=True)                      # [B,A,1]

    return np.where(count == 0, 0.0, summed / np.maximum(count, 1))


import numpy as np


def nearest_valid_agent(ego, agents, validity):
    """
    Args
    ----
    ego      : np.ndarray, shape [B, 2]
    agents   : np.ndarray, shape [B, A, 2]
    validity : np.ndarray (bool), shape [B, A]

    Returns
    -------
    np.ndarray (int), shape [B] — index of nearest valid agent, or -1.
    """
    # TODO: pairwise distances via broadcasting, set masked entries to +inf before argmin,
    # then fix up batches with no valid agents.

    ego_b = ego[:, None, :]

    any_valid = np.any(validity, axis=-1)

    distance = np.where(validity, np.sum((ego_b - agents)**2, axis=-1), 1e10)

    return np.where(any_valid, np.argmin(distance, axis=-1), -1)


def nearest_k_valid_agents(
    ego: np.ndarray, agents: np.ndarray, validity: np.ndarray, k: int
) -> np.ndarray:
    """
    Args
    ----
    ego      : np.ndarray, shape [B, 2]
    agents   : np.ndarray, shape [B, A, 2]
    validity : np.ndarray (bool), shape [B, A]
    k        : int — how many neighbors to return

    Returns
    -------
    np.ndarray (int), shape [B, k] — indices of the k nearest valid agents,
    sorted ascending by distance. Slots without a valid agent are filled with -1.
    """
    ego_b = ego[:, None, :]
    dist_sq = np.sum((ego_b - agents) ** 2, axis=-1)         # [B, A]
    masked_dist = np.where(validity, dist_sq, np.inf)         # [B, A]

    k = min(k, agents.shape[1])                               # clamp if k > A
    top_k_idx = np.argsort(masked_dist, axis=-1)[:, :k]       # [B, k]

    top_k_dist = np.take_along_axis(masked_dist, top_k_idx, axis=-1)
    return np.where(np.isinf(top_k_dist), -1, top_k_idx)


def _top_k_ade(preds: np.ndarray, conf: np.ndarray, gt: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Shared gather pipeline for the minADE / weighted-ADE family.

    Returns
    -------
    ade      : np.ndarray, shape [B, k] — ADE per top-k prediction
    top_conf : np.ndarray, shape [B, k] — confidence scores of those top-k preds
    """
    actual_k = min(k, conf.shape[-1])
    conf_idx = np.argpartition(-conf, kth=actual_k - 1, axis=-1)[:, :actual_k]   # [B, k]
    pred_top_k = np.take_along_axis(preds, conf_idx[..., None, None], axis=1)    # [B, k, T, 2]

    dist = np.sqrt(np.sum((pred_top_k - gt[:, None, :, :]) ** 2, axis=-1))       # [B, k, T]
    ade = np.mean(dist, axis=-1)                                                  # [B, k]
    top_conf = np.take_along_axis(conf, conf_idx, axis=-1)                       # [B, k]
    return ade, top_conf


def min_ade_top_k(preds: np.ndarray, conf: np.ndarray, gt: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Canonical minADE@K: minimum ADE over the top-k most confident predictions.

    Args
    ----
    preds : np.ndarray, shape [B, K, T, 2]
    conf  : np.ndarray, shape [B, K]
    gt    : np.ndarray, shape [B, T, 2]
    k     : int, number of top predictions to consider

    Returns
    -------
    np.ndarray, shape [B] — min ADE over the top-k most confident preds.
    """
    ade, _ = _top_k_ade(preds, conf, gt, k)
    return ade.min(axis=-1)


def weighted_ade_top_k(preds: np.ndarray, conf: np.ndarray, gt: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Confidence-weighted ADE over the top-k most confident predictions.
    Weights are the softmax of the top-k confidence scores.

    Args
    ----
    preds : np.ndarray, shape [B, K, T, 2]
    conf  : np.ndarray, shape [B, K]
    gt    : np.ndarray, shape [B, T, 2]
    k     : int, number of top predictions to consider

    Returns
    -------
    np.ndarray, shape [B] — Σ softmax(top_conf) * ADE over the top-k preds.
    """
    ade, top_conf = _top_k_ade(preds, conf, gt, k)

    shifted = top_conf - top_conf.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights = weights / weights.sum(axis=-1, keepdims=True)

    return (weights * ade).sum(axis=-1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """
    Args
    ----
    boxes      : np.ndarray, shape [N, 4], (x1, y1, x2, y2)
    scores     : np.ndarray, shape [N]
    iou_thresh : float

    Returns
    -------
    np.ndarray (int), shape [K] — kept indices, score-descending.
    """
    def calc_area(box) -> np.ndarray:
        return np.maximum(0, (box[:, 2] - box[:, 0])) * np.maximum(0, (box[:, 3] - box[:, 1]))

    def calc_iou(box_a, box_b):
        area_a = calc_area(box_a)
        area_b = calc_area(box_b)

        bl = np.maximum(box_a[:, :2], box_b[:, :2])
        tr = np.minimum(box_a[:, 2:], box_b[:, 2:])
        wh = np.clip(tr - bl, a_min=0, a_max=None)
        intersection = wh[:, 0] * wh[:, 1]

        return intersection / (area_a + area_b - intersection + 1e-8)

    if scores.shape[0] == 0:
        return np.array([], dtype=np.int64)

    mask = np.full_like(scores, True, dtype=bool)
    kept_indices = []

    while np.any(mask):
        max_idx = np.argmax(np.where(mask, scores, -np.inf))
        kept_indices.append(max_idx)
        mask[max_idx] = False

        box = boxes[max_idx]
        remain_idx = np.flatnonzero(mask)
        iou = calc_iou(box[None, :], boxes[remain_idx])
        iou_above = np.flatnonzero(iou >= iou_thresh)
        mask[remain_idx[iou_above]] = False

    return np.array(kept_indices, dtype=np.int64)
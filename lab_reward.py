"""Pure-Python IoU reward and quick unit checks for optional lab experiment E5."""

from __future__ import annotations

from scripts.score import iou_xyxy, parse_output


def compute_iou_reward(completion: str, gt_boxes: list[dict]) -> float:
    """Compute the lab's verified reward for normalized 0–1000 boxes.

    Ground-truth boxes use ``{"label": ..., "bbox_2d": [x1,y1,x2,y2]}``.
    Invalid JSON receives zero; two empty lists receive one. Otherwise boxes
    are greedily matched by descending IoU, same class, and IoU >= 0.5.
    """
    predictions = parse_output(completion)
    if predictions is None:
        return 0.0
    if not gt_boxes and not predictions:
        return 1.0

    pairs = []
    for gt_index, ground_truth in enumerate(gt_boxes):
        gt_box = ground_truth.get("bbox_2d", ground_truth.get("bbox"))
        for pred_index, prediction in enumerate(predictions):
            if ground_truth["label"] == prediction["label"]:
                pairs.append((iou_xyxy(gt_box, prediction["bbox"]), gt_index, pred_index))
    pairs.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    total_iou = 0.0
    for iou, gt_index, pred_index in pairs:
        if iou < 0.5:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        total_iou += iou

    return 0.2 + 0.8 * total_iou / max(len(gt_boxes), len(predictions), 1)


def check_iou_reward() -> None:
    """Check perfect, malformed, wrong-class, extra-box, and empty cases."""
    gt = [{"label": "pothole", "bbox_2d": [100, 100, 200, 200]}]
    assert compute_iou_reward('[{"label":"pothole","bbox_2d":[100,100,200,200]}]', gt) == 1.0
    assert compute_iou_reward("not json", gt) == 0.0
    wrong_class = compute_iou_reward('[{"label":"crack","bbox_2d":[100,100,200,200]}]', gt)
    assert abs(wrong_class - 0.2) < 1e-9
    extra_box = compute_iou_reward(
        '[{"label":"pothole","bbox_2d":[100,100,200,200]},'
        '{"label":"pothole","bbox_2d":[400,400,500,500]}]',
        gt,
    )
    assert abs(extra_box - 0.6) < 1e-9
    assert compute_iou_reward("[]", []) == 1.0
    print("E5 reward checks passed: perfect, malformed, wrong class, extra box, both empty")


if __name__ == "__main__":
    check_iou_reward()

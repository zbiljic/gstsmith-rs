use std::cmp::Ordering;

use super::decode::Detection;

pub(super) fn suppress(candidates: &mut Vec<Detection>, iou_threshold: f32, max_detections: usize) {
    if max_detections == 0 {
        candidates.clear();
        return;
    }
    candidates.sort_unstable_by(detection_order);
    let mut retained = 0usize;
    for index in 0..candidates.len() {
        let Some(candidate) = candidates.get(index).copied() else {
            break;
        };
        if candidates.get(..retained).is_some_and(|previous| {
            previous.iter().any(|previous| {
                previous.class == candidate.class
                    && intersection_over_union(previous, &candidate) > iou_threshold
            })
        }) {
            continue;
        }
        let Some(slot) = candidates.get_mut(retained) else {
            break;
        };
        *slot = candidate;
        retained += 1;
        if retained == max_detections {
            break;
        }
    }
    candidates.truncate(retained);
}

fn detection_order(left: &Detection, right: &Detection) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.class.cmp(&right.class))
        .then_with(|| left.ordinal.cmp(&right.ordinal))
}

fn intersection_over_union(left: &Detection, right: &Detection) -> f32 {
    let intersection_width = (left.x2.min(right.x2) - left.x1.max(right.x1)).max(0.0);
    let intersection_height = (left.y2.min(right.y2) - left.y1.max(right.y1)).max(0.0);
    let intersection = intersection_width * intersection_height;
    let left_area = (left.x2 - left.x1) * (left.y2 - left.y1);
    let right_area = (right.x2 - right.x1) * (right.y2 - right.y1);
    let union = left_area + right_area - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

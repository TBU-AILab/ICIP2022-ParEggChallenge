import numpy as np


def nms(
    predictions: list,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """
    # we extract coordinates for every
    # prediction box present in P
    x1 = np.array([])
    y1 = np.array([])
    x2 = np.array([])
    y2 = np.array([])
    scores = np.array([])
    for ann in predictions:
        x1 = np.append(x1, ann['bbox'][0])
        y1 = np.append(y1, ann['bbox'][1])
        x2 = np.append(x2, ann['bbox'][0]+ann['bbox'][2])
        y2 = np.append(y2, ann['bbox'][1]+ann['bbox'][3])
        scores = np.append(scores, ann['score'])

    # calculate area of every block in P
    areas = (x2-x1) * (y2-y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = x1[order]
        xx2 = x2[order]
        yy1 = y1[order]
        yy2 = y2[order]

        # find the coordinates of the intersection boxes
        xx1 = np.maximum(xx1, x1[idx])
        yy1 = np.maximum(yy1, y1[idx])
        xx2 = np.minimum(xx2, x2[idx])
        yy2 = np.minimum(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = np.clip(w, 0, None)
        h = np.clip(h, 0, None)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = areas[order]

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = np.minimum(rem_areas, areas[idx])
            # find the IoU of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # keep the boxes with IoU less than thresh_iou
        mask = match_metric_value < match_threshold
        order = order[mask]
    return [predictions[i] for i in keep]


def mean_nms(
    predictions: list,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply non-maximum like suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    The bbox coordinates are averaged rather than select only the one with the highest confidence.
    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """
    # we extract coordinates for every
    # prediction box present in P
    x1 = np.array([])
    y1 = np.array([])
    x2 = np.array([])
    y2 = np.array([])
    scores = np.array([])
    for ann in predictions:
        x1 = np.append(x1, ann['bbox'][0])
        y1 = np.append(y1, ann['bbox'][1])
        x2 = np.append(x2, ann['bbox'][0]+ann['bbox'][2])
        y2 = np.append(y2, ann['bbox'][1]+ann['bbox'][3])
        scores = np.append(scores, ann['score'])

    # calculate area of every block in P
    areas = (x2-x1) * (y2-y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []
    to_merge = []

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = x1[order]
        xx2 = x2[order]
        yy1 = y1[order]
        yy2 = y2[order]

        # find the coordinates of the intersection boxes
        xx1 = np.maximum(xx1, x1[idx])
        yy1 = np.maximum(yy1, y1[idx])
        xx2 = np.minimum(xx2, x2[idx])
        yy2 = np.minimum(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = np.clip(w, 0, None)
        h = np.clip(h, 0, None)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = areas[order]

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = np.minimum(rem_areas, areas[idx])
            # find the IoU of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # choose the boxes that would be eliminated otherwise
        mask2 = match_metric_value > match_threshold
        to_merge.append(order[mask2].tolist())

        # keep the boxes with IoU less than thresh_iou
        mask = match_metric_value < match_threshold
        order = order[mask]
    best_bboxes = [predictions[i] for i in keep]
    bboxes_to_merge = []
    for ind_bbox in to_merge:
        bboxes_to_merge.append([predictions[i] for i in ind_bbox])

    best_pred = mean_bbox_coordinates(best_bboxes, bboxes_to_merge)
    return best_pred


def mean_bbox_coordinates(best_bboxes, bboxes_to_merge):

    for best_pred, preds_to_merge in zip(best_bboxes, bboxes_to_merge):
        bbox = best_pred['bbox']
        for pred_to_merge in preds_to_merge:
            bbox = [bbox[i]+pred_to_merge['bbox'][i] for i in range(len(bbox))]
        bbox = [bbox[i]/(len(preds_to_merge)+1) for i in range(len(bbox))]
        diff = [abs(best_pred['bbox'][i]-bbox[i]) for i in range(len(bbox))]
        if diff < [2500., 2500., 2500., 2500.]:
            best_pred['bbox'] = bbox

    return best_bboxes


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def get_img_ids(results):
    img_ids = []
    for res in results:
        if res['image_id'] not in img_ids:
            img_ids.append(res['image_id'])
    return img_ids


def pd_dict_to_results(pd_dict):
    results = []
    for key in pd_dict['image_id'].keys():
        result = {}
        for name in pd_dict:
            result[name] = pd_dict[name][key]
        results.append(result)
    return results


def separate_image_results(img_ids, results):
    sorted_results = []
    for img_id in img_ids:
        sorted_results.append([ann for ann in results if ann['image_id'] == img_id])
    return sorted_results


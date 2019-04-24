"""THis is helper function about IOU computation
====
Features:
> compute iou over two polygons
"""

import numpy as np
from skimage.draw import polygon

def compute_polygons_iou(polygon1, polygon2):
    """
    Intersection over union between two shapely polygons.
    """
    x1 = np.array(polygon1[0::2])
    y1 = np.array(polygon1[1::2])
    x2 = np.array(polygon2[0::2])
    y2 = np.array(polygon2[1::2])
    IoU = iou(x1, y1, x2, y2)
    return IoU

def approx_area_of_intersection(dt_x, dt_y, gt_x, gt_y):
    """
    This helper determine if both polygons are intersecting with each others with an approximation method.
    Area of intersection represented by the minimum bounding rectangular [xmin, ymin, xmax, ymax]
    """
    dt_ymax = np.max(dt_y)
    dt_xmax = np.max(dt_x)
    dt_ymin = np.min(dt_y)
    dt_xmin = np.min(dt_x)

    gt_ymax = np.max(gt_y)
    gt_xmax = np.max(gt_x)
    gt_ymin = np.min(gt_y)
    gt_xmin = np.min(gt_x)

    all_min_ymax = np.minimum(dt_ymax, gt_ymax)
    all_max_ymin = np.maximum(dt_ymin, gt_ymin)

    intersect_heights = np.maximum(0.0, (all_min_ymax - all_max_ymin))

    all_min_xmax = np.minimum(dt_xmax, gt_xmax)
    all_max_xmin = np.maximum(dt_xmin, gt_xmin)
    intersect_widths = np.maximum(0.0, (all_min_xmax - all_max_xmin))

    return intersect_heights * intersect_widths

def iou(dt_x, dt_y, gt_x, gt_y):
    """
    This helper determine the intersection over union of two polygons.
    """

    if approx_area_of_intersection(dt_x, dt_y, gt_x, gt_y) > 1: # only proceed if it passes the approximation test
        ymax = np.maximum(np.max(dt_y), np.max(gt_y)) + 1
        xmax = np.maximum(np.max(dt_x), np.max(gt_x)) + 1
        bin_mask = np.zeros((ymax, xmax))
        dt_bin_mask = np.zeros_like(bin_mask)
        gt_bin_mask = np.zeros_like(bin_mask)

        rr, cc = polygon(dt_y, dt_x)
        dt_bin_mask[rr, cc] = 1

        rr, cc = polygon(gt_y, gt_x)
        gt_bin_mask[rr, cc] = 1

        final_bin_mask = dt_bin_mask + gt_bin_mask

        inter_map = np.where(final_bin_mask == 2, 1, 0)
        inter = np.sum(inter_map)

        union_map = np.where(final_bin_mask > 0, 1, 0)
        union = np.sum(union_map)
        return inter / float(union + 1.0)
    else:
        return 0
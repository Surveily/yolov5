"""
Copyright (c) Surveily. All rights reserved.
"""

from collections import namedtuple
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class YoloDetectionsMatchingService:

    def __init__(self) -> None:
        pass

    def find_detections_assignment(self, targets: np.ndarray, detections: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cost_matrix = self.__calculate_cost_matrix(
            targets, detections)

        rows_assignment, cols_assignments = linear_sum_assignment(
            -cost_matrix)

        return rows_assignment, cols_assignments, cost_matrix

    def __calculate_cost_matrix(self, targets: np.ndarray, detections: np.ndarray) -> np.ndarray:
        cost_matrix = np.zeros((targets.shape[0], detections.shape[0]))
        for targets_index in range(targets.shape[0]):
            for detections_index in range(detections.shape[0]):
                cost_matrix[targets_index][detections_index] = self.__calculate_iou_yolo(
                    targets[targets_index], detections[detections_index])

        return cost_matrix

    def __calculate_iou_yolo(self, target: np.ndarray, detection: np.ndarray) -> float:
        Rect = namedtuple("Rect", "x y width height")
        lhs = Rect(target[2], target[3], target[4], target[5])
        rhs = Rect(detection[2], detection[3], detection[4], detection[5])

        intersection_width = min(
            lhs.x + lhs.width, rhs.x + rhs.width) - max(lhs.x, rhs.x)
        intersection_height = min(
            lhs.y + lhs.height, rhs.y + rhs.height) - max(lhs.y, rhs.y)

        if intersection_width <= 0 or intersection_height <= 0:
            return 0

        intersection = intersection_width * intersection_height
        union = ((lhs.width * lhs.height) +
                 (rhs.width * rhs.height)) - intersection

        return intersection / union

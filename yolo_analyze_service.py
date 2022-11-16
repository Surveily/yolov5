from typing import List

import numpy as np

from yolo_detections_matching_service import YoloDetectionsMatchingService


class YoloAnalyzeService:
    def __init__(self, minimum_iou: float, minimum_confidence: float) -> None:
        self.__minimum_iou = minimum_iou
        self.__minimum_confidence = minimum_confidence
        self.__detections_matching_service = YoloDetectionsMatchingService()

    def analyze_batch(self, targets: np.ndarray, detections: np.ndarray):
        wrong_detections = []
        not_detected_targets = []
        wrong_detections_labels = []

        targets_batched = self.__parse_targets(targets)
        detections_batched = self.__parse_targets(detections)

        for targets_batch, detections_batch in zip(targets_batched, detections_batched):
            rows_assignment, columns_assignments, cost_matrix = self.__detections_matching_service.find_detections_assignment(
                targets_batch, detections_batch)

            found_targets = set()
            matched_detections = set()

            for row, col in zip(rows_assignment, columns_assignments):
                iou = cost_matrix[row][col]

                if iou < self.__minimum_iou:
                    continue

                found_targets.add(row)
                matched_detections.add(col)

                target_label = targets_batch[row][1]
                detection_label = detections_batch[col][1]

                if target_label != detection_label:
                    wrong_detections_labels.append(detections_batch[col])
                    continue

            for target_index in range(targets_batch.shape[0]):
                if not found_targets.__contains__(target_index):
                    not_detected_targets.append(targets_batch[target_index])

            for detection_index in range(detections_batch.shape[0]):
                if not matched_detections.__contains__(detection_index):
                    wrong_detections.append(detections_batch[detection_index])

        all_mistakes = wrong_detections + wrong_detections_labels + not_detected_targets

        if all_mistakes == []:
            return np.array([])
        return np.concatenate(all_mistakes).reshape(-1, 7)

    def __parse_targets(self, targets: np.ndarray) -> List[np.ndarray]:
        targets_batched = []
        last_batch_index = -1

        for target in targets:
            if target.shape[0] == 6:
                target = np.append(target, [1.0])

            if target[6] < self.__minimum_confidence:
                continue

            batch_index = int(target[0])

            while batch_index != last_batch_index:
                targets_batched.append([])
                last_batch_index += 1

            targets_batched[batch_index].append(target)

        targets_final = []

        for targets_batch in targets_batched:
            if targets_batch == []:
                targets_final.append(np.array([]))
                continue
            target_final = np.concatenate(targets_batch)
            target_final = target_final.reshape(-1, 7)
            targets_final.append(target_final)
        return targets_final

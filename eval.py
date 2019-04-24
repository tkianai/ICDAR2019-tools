"""This is the eval tools for ICDAR2019 competitions
====
Features:
> mAP evaluation by cocoAPI
> F score evaluation
====
Author:
> tkianai
"""

import os
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from imantics import Mask as maskToPolygon
from utils.iou import compute_polygons_iou

class IcdarEval(object):
    def __init__(self, dt_file, gt_file=None, iou_threshold=0.5):
        self.gt_file = gt_file
        self.dt_file = dt_file
        self.iou_threshold = iou_threshold
        self.dt_anns = json.load(open(dt_file))
        if gt_file is None:
            self.gt_anns = None
        else:
            self.gt_anns = json.load(open(gt_file))

        self.Precision = None
        self.Recall = None
    
    def calculate_F_score(self, Precision, Recall):
        eps = 1e-7
        F_score = 2.0 * Precision * Recall / (Precision + Recall + eps)
        return F_score

    def eval_map(self, mode='segm'):
        """evaluate mean average precision
        
        Keyword Arguments:
            mode {str} -- could be choose from ['bbox' | 'segm'] (default: {'segm'})
        """
        if mode not in ['bbox', 'segm']:
            raise NotImplementedError("Mode [{}] doesn't been implemented, choose from [bbox, segm]!".format(mode))

        # eval map
        Gt = COCO(self.gt_file)
        Dt = Gt.loadRes(self.dt_file)

        evalObj = COCOeval(Gt, Dt, mode)
        imgIds = sorted(Gt.getImgIds())
        evalObj.params.imgIds = imgIds
        evalObj.evaluate()
        evalObj.accumulate()
        evalObj.summarize()

    def save_pr_curve(self, save_name='./data/P_R_curve.png'):
        if self.Precision is None or self.Recall is None:
            return

        # save the P-R curve
        save_dir = os.path.dirname(save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.clf()
        plt.plot(self.Recall, self.Precision)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Precision-Recall Curve')
        plt.grid()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(save_name, dpi=400)
        print('Precision-recall curve has been written to {}'.format(save_name))

    def eval_F_by_coco(self, threshold=None, mode='segm'):
        iou_threshold = self.iou_threshold if threshold is None else threshold
        assert iou_threshold >=0.0 and iou_threshold <= 1.0, "The IOU threshold [{}] is illegal!".format(iou_threshold)
        if mode not in ['bbox', 'segm']:
            raise NotImplementedError("Mode [{}] doesn't been implemented, choose from [bbox, segm]!".format(mode))

        # eval map
        Gt = COCO(self.gt_file)
        Dt = Gt.loadRes(self.dt_file)

        evalObj = COCOeval(Gt, Dt, mode)
        imgIds = sorted(Gt.getImgIds())
        evalObj.params.imgIds = imgIds
        evalObj.params.iouThrs = [iou_threshold]
        evalObj.params.areaRng = [[0, 10000000000.0]]
        evalObj.params.maxDets = [100]

        evalObj.evaluate()
        evalObj.accumulate()

        Precision = evalObj.eval['precision'][0, :, 0, 0, 0]
        Recall = evalObj.params.recThrs
        Scores = evalObj.eval['scores'][0, :, 0, 0, 0]

        F_score = self.calculate_F_score(Precision, Recall)

        # calculate highest F score
        idx = np.argmax(F_score)
        results = dict(
            F_score=F_score[idx],
            Precision=Precision[idx],
            Recall=Recall[idx],
            score=Scores[idx],
        )

        # summarize
        print('---------------------- F1 ---------------------- ')
        print('Maximum F-score: %f' % results['F_score'])
        print('  |-- Precision: %f' % results['Precision'])
        print('  |-- Recall   : %f' % results['Recall'])
        print('  |-- Score    : %f' % results['score'])
        print('------------------------------------------------ ')

        self.Precision = Precision
        self.Recall = Recall

    def eval_F(self, threshold=None):
        iou_threshold = self.iou_threshold if threshold is None else threshold
        assert iou_threshold >=0.0 and iou_threshold <= 1.0, "The IOU threshold [{}] is illegal!".format(iou_threshold)
        assert self.gt_anns is not None, "GroundTruth must be needed!"
        
        gt_polygons_number = 0
        gt_polygons_set = {}

        gt_annotations = self.gt_anns['annotations']
        gt_imgIds = set()
        for itm in gt_annotations:
            gt_imgIds.add(itm['image_id'])
        
        for imgId in gt_imgIds:
            polygons = []
            for itm in gt_annotations:
                if itm['image_id'] == imgId:
                    # polygons
                    gt_segm = np.array(itm['segmentation']).ravel().tolist()
                    polygons.append(gt_segm)
            
            gt_polygons_number += len(polygons)
            gt_polygons_set[imgId] = polygons

        dt_gt_match_all = []
        dt_scores_all = []
        
        dt_imgIds = set()
        dt_annotations = self.dt_anns
        for itm in dt_annotations:
            dt_imgIds.add(itm['image_id'])

        for imgId in tqdm(dt_imgIds):
            if imgId not in gt_polygons_set:
                print("Image ID [{}] not found in GroundTruth file, this will be ignored!".format(imgId))
                continue
            
            gt_polygons = gt_polygons_set[imgId]
            dt_polygons = []
            dt_scores = []

            for itm in dt_annotations:
                if itm['image_id'] == imgId:
                    # mask
                    _mask = maskUtils.decode(itm['segmentation']).astype(np.bool)
                    polygons = maskToPolygon(_mask).polygons()
                    roi_areas = [cv2.contourArea(points) for points in polygons.points]
                    idx = roi_areas.index(max(roi_areas))
                    dt_polygons.append(polygons.points[idx].tolist())
                    dt_scores.append(itm['score'])
            
            dt_gt_match = []
            # TODO: methods of match should be optimizied according to LSVT requirements
            for dt_polygon in dt_polygons:
                match_flag = False
                for gt_polygon in gt_polygons:
                    if compute_polygons_iou(dt_polygon, gt_polygon) >= iou_threshold:
                        match_flag = True
                        break
                dt_gt_match.append(match_flag)
            dt_gt_match_all.extend(dt_gt_match)
            dt_scores_all.extend(dt_scores)

        assert len(dt_gt_match_all) == len(dt_scores_all), "each polygon should have it's score!"

        # calculate precision, recall and F score under different score threshold
        dt_gt_match_all = np.array(dt_gt_match_all, dtype=np.bool).astype(np.int)
        dt_scores_all = np.array(dt_scores_all)

        # sort according to score
        sort_idx = np.argsort(dt_scores_all)[::-1]
        dt_gt_match_all = dt_gt_match_all[sort_idx]
        dt_scores_all = dt_scores_all[sort_idx]

        number_positive = np.cumsum(dt_gt_match_all)
        number_detected = np.arange(1, len(dt_gt_match_all) + 1)
        Precision = number_positive.astype(np.float) / number_detected.astype(np.float)
        Recall = number_positive.astype(np.float) / float(gt_polygons_number)
        F_score = self.calculate_F_score(Precision, Recall)

        # calculate highest F score
        idx = np.argmax(F_score)
        results = dict(
            F_score=F_score[idx],
            Precision=Precision[idx],
            Recall=Recall[idx],
            score=dt_scores_all[idx],
        )

        # summarize
        print('---------------------- F1 ---------------------- ')
        print('Maximum F-score: %f' % results['F_score'])
        print('  |-- Precision: %f' % results['Precision'])
        print('  |-- Recall   : %f' % results['Recall'])
        print('  |-- Score    : %f' % results['score'])
        print('------------------------------------------------ ')

        self.Precision = Precision
        self.Recall = Recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mAP evaluation on ICDAR2019')
    parser.add_argument('--gt-file', default='data/gt.json', type=str, help='annotation | groundtruth file.')
    parser.add_argument('--dt-file', default='data/dt.json', type=str, help='detection results of coco annotation style.')
    args = parser.parse_args()
    # judge file existence
    if not os.path.exists(args.gt_file):
        print("File Not Found Error: {}".format(args.gt_file))
        exit(404)
    if not os.path.exists(args.dt_file):
        print("File Not Found Error: {}".format(args.dt_file))
        exit(404)
    eval_icdar = IcdarEval(args.dt_file, args.gt_file)
    eval_icdar.eval_map(mode='bbox')
    eval_icdar.eval_map(mode='segm')
    eval_icdar.eval_F_by_coco(threshold=0.5, mode='segm')
    eval_icdar.save_pr_curve()
    # eval_icdar.eval_F()

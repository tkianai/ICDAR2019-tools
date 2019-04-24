"""Transfer the detection results of coco-style to submission format
====
Features:
> LSVT
> ReCTS
====
Author:
> tkianai
"""

import os
import json
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import pycocotools.mask as maskUtils
from imantics import Mask as maskToPolygon

CONFIG = {
    "lsvt": {
        "name": "lsvt",
        "TOTAL_NUM": 20000,
        "START": 0,
        "id_prefix": "res_",
        "id_form": "{}",
        "id_suffix": "",
        "file_suffix": 'json',
        "points_len": None,
    },
    "rects": {
        "name": "rects",
        "TOTAL_NUM": 5000,
        "START": 1,
        "id_prefix": "test_",
        "id_form": "{:0>6d}",
        "id_suffix": ".jpg",
        "file_suffix": 'txt',
        "points_len": 4
    },
    "art": {

    }
}

class FORMAT(object):
    def __init__(self, dt_file, mode='lsvt'):
        mode = mode.lower()
        assert mode in ['lsvt', 'rects'], "Mode [{}] is not supported! Try [lsvt | rects]!"
        self.results = json.load(open(dt_file))
        self.config = CONFIG[mode]
    
    def check_clockwise(self, points):
        points = np.array(points)
        points = cv2.convexHull(points)
        points = points[:, 0, :]
        y = points[:, 1]
        idx = np.argmax(y)
        x1 = points[(idx - 1 + len(points)) % len(points)]
        x2 = points[idx]
        x3 = points[(idx + 1) % len(points)]
        x2_x1 = x2 - x1
        x3_x2 = x3 - x2
        judge_result = x2_x1[0] * x3_x2[1] - x2_x1[1] * x3_x2[0]
        if judge_result < 0:
            points = points[::-1]
        
        return points.tolist()

    def check_points_len(self, points, bbox):
        if len(points) == self.config['points_len']:
            return points

        roi_area = cv2.contourArea(points)
        thr = 0.03
        while thr < 0.08:
            points_validate = []
            idx_remove = []
            for p in range(len(points)):
                index = list(range(len(points)))
                index.remove(p)
                for k in idx_remove:
                    index.remove(k)
                area = cv2.contourArea(points[index])
                if np.abs(roi_area - area) / roi_area > thr:
                    points_validate.append(points[p])
                else:
                    idx_remove.append(p)
            if len(points_validate) == self.config['points_len']:
                return np.array(points_validate)

            thr += 0.01
        
        # return minAreaRect
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box[box[:, 0] < bbox[0], 0] = bbox[0]
        box[box[:, 1] < bbox[1], 1] = bbox[1]
        box[box[:, 0] > bbox[0] + bbox[2], 0] = bbox[0] + bbox[2]
        box[box[:, 1] > bbox[1] + bbox[3], 1] = bbox[1] + bbox[3]

        return box.astype(np.int)
        

    def format(self, save_name='./data/lsvt_submission.json'):
        
        save_dir = os.path.dirname(save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.basename(save_name)
        filename = filename.split('.')
        filename[-1] = self.config['file_suffix']
        filename = '.'.join(filename)
        save_name = os.path.join(save_dir, filename)

        submission_out = {}
        for itm in tqdm(self.results):
            fileid = self.config['id_prefix'] + itm['image_id'] + self.config['id_suffix']
            if fileid not in submission_out:
                submission_out[fileid] = []
            
            _mask = maskUtils.decode(itm['segmentation']).astype(np.bool)
            polygons = maskToPolygon(_mask).polygons()
            roi_areas = [cv2.contourArea(points) for points in polygons.points]
            if max(roi_areas) < 1:
                continue
            idx = roi_areas.index(max(roi_areas))
            points = polygons.points[idx]
            roi_area = roi_areas[idx]

            # eliminate unnecessarily points
            points_validate = []
            idx_remove = []
            for p in range(len(points)):
                index = list(range(len(points)))
                index.remove(p)
                for k in idx_remove:
                    index.remove(k)
                area = cv2.contourArea(points[index])
                if np.abs(roi_area - area) / roi_area > 0.02:
                    points_validate.append(points[p])
                else:
                    idx_remove.append(p)
            points_validate = np.array(points_validate)

            if self.config['points_len'] is not None:
                points_validate = self.check_points_len(points_validate, itm['bbox'])

            points_validate = self.check_clockwise(points_validate.tolist())

            info = {}
            info['points'] = points_validate
            info['confidence'] = float("{:.3f}".format(itm['score']))
            submission_out[fileid].append(info)

        # validate all files
        for i in range(self.config['START'], self.config['START'] + self.config['TOTAL_NUM']):
            id_ = self.config['id_prefix'] + self.config['id_form'].format(i) + self.config['id_suffix']
            if id_ not in submission_out:
                submission_out[id_] = []

        # save file
        with open(save_name, 'w') as w_obj:
            if self.config['file_suffix'] == 'json':
                json.dump(submission_out, w_obj)
            else:
                files = sorted(submission_out.keys())
                for file in files:
                    w_obj.write("{}\n".format(file))
                    for det_res in submission_out[file]:
                        points = []
                        for tmp_point in det_res['points']:
                            points.append(str(int(tmp_point[0])))
                            points.append(str(int(tmp_point[1])))
                        points = ','.join(points)
                        w_obj.write(points + '\n')

        print("Results have been saved to {}".format(save_name))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Tranfer the format to submission style.")
    parser.add_argument('--dt-file', default=None, help='coco style detection file.')
    parser.add_argument('--mode', default='lsvt', help='choose the format you want transfer to, [lsvt | rects].')
    parser.add_argument('--save', default='data/lsvt_task1.json', help='filepath to save the transfered results.')
    args = parser.parse_args()

    formatObj = FORMAT(args.dt_file, mode=args.mode)
    formatObj.format(args.save)
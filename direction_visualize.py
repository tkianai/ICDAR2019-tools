
import json
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='visulize points extending direction.')
parser.add_argument('--json')
parser.add_argument('--key')
args = parser.parse_args()
data = json.load(open(args.json))
sample = data[args.key]
for itm in sample[2:3]:
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for point in itm['points']:
        image = cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 255, 0), 1)
        cv2.imshow("direction", image)
        cv2.waitKey()


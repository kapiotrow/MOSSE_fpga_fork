from os.path import join
import os
import cv2
import argparse
import numpy as np
import vot
import sys
import json

from deep_mosse import DeepMosse

CONFIG = '/configs/config.json'
with open(CONFIG, 'r') as json_file:
    config = json.load(json_file)

handle = vot.VOT("rectangle")
selection = handle.region()
selection = [selection.x, selection.y, selection.width, selection.height]

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
tracker = DeepMosse(image, selection, config)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region = tracker.track(image)
    region = vot.Rectangle(region[0], region[1], region[2], region[3])
    handle.report(region)
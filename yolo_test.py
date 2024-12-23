#!/usr/bin/env python3
from ultralytics import YOLO
import os
import numpy as np

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

model = YOLO(my_absolute_dirpath + "/runs/detect/train13/weights/best.pt")
# model = YOLO("yolo11n.pt")

# results = model.train(data= my_absolute_dirpath + "/datasets/tank2/data.yaml", epochs=100, imgsz=720)

# metrics = model.val()
# metrics.box.map
# metrics.box.map50
# metrics.box.map75
# metrics.box.maps

results = model("tank.png")


for result in results:
    names = result.names
    print(names[0])
    result.show()
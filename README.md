acknowledgements:

https://github.com/TianhongDai/mosse-object-tracking/blob/master/examples/surfer.gif
https://github.com/pytorch/examples/tree/master/imagenet

how to:

1. requirements:git a
    - opencv-python
    - torch
    - torchvision
    - fxpmath
    - imutils
    - brevitas
    - easydict

2. demo on vot dataset
    - pip install vot-toolkit
    - create a dataset workspace in this repo <workspace-path>:
    - ```vot initialize vot2015 --workspace <workspace-path>`` 
    - to run a quick demo: ```python evaluate_VOT.py --sequences <workspace-path>/sequences/ --seq tiger --debug```
    - to run on a custom sequence, add a new sequence with ```color``` directory and a ```groundtruth.txt``` file with bounding boxes per frame in format:
        ```X1,Y1,X2,Y2,X3,Y3,X4,Y4```, where Xi and Yi are the coordinates of corner i of the bounding box in frame N, the N-th row in the text file. Just the first frame is enough. 

3. vot evaluation (official guide https://www.votchallenge.net/howto/ > Toolkit overview)

    
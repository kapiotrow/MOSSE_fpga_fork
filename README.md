acknowledgements:

https://github.com/TianhongDai/mosse-object-tracking/blob/master/examples/surfer.gif
https://github.com/pytorch/examples/tree/master/imagenet

how to:

1. requirements:
    - opencv-python
    - torch
    - torchvision
    - fxpmath
    - imutils
    - brevitas

2. vot evaluation (official guide https://www.votchallenge.net/howto/ > Toolkit overview)
    - pip install vot-toolkit
    - create dataset workspace in desired <workspace-path>:
    - ```vot initialize vot2015 --workspace <workspace-path>`` // You can get the list of included dataset stacks using vot initialize (without the "vot2015")

import torchvision.transforms as transforms
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from imagenet.finn_models import get_finnlayer


def cnn_preprocess(data):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        result = transform(data)
        result = result.unsqueeze(dim=0)
        result = result.to(torch.device('cuda:0'))

        return result

pad = 2
backbone = get_finnlayer('output/finnlayer_weights/savegame_0_15000.pth.tar', strict=False)
backbone.to(torch.device('cuda:0'))
test_img = cv2.imread('test_input.ppm')
test_img = cv2.copyMakeBorder(test_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

# test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_input = cnn_preprocess(test_img)
test_output = backbone(test_input)


# x = np.range()
# print('x shape:', x.shape)
# conv = torch.nn.Conv2d(3, 1, kernel_size=3, stride=3, bias=False)
# conv.weight = torch.nn.Parameter(torch.ones((1, 1, 3, 3)))
# # conv.bias = torch.nn.Parameter(torch.tensor(0.69).reshape((1)))
# print(conv.weight)
# pool = torch.nn.MaxPool2d((3, 3))
# y = conv(torch.tensor(x).float())
# print(y.shape)

# print(test_img.shape)
# np.savetxt('R.csv', test_img[:, :, 0].astype(np.uint8))
# print(test_img[213:, :10, 0])
# print(test_img[213:, :10, 1])
# print(test_img[213:, :10, 2])
# plt.imshow(test_img)
# plt.show()
# print(test_output[0, 0])
# print('min:', test_output.min())
import torchvision.transforms as transforms
import cv2
import torch

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


backbone = get_finnlayer('output/finnlayer_weights/savegame_0_15000.pth.tar', strict=False)
backbone.to(torch.device('cuda:0'))
test_img = cv2.imread('test_input.ppm')
# test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_input = cnn_preprocess(test_img)
test_output = backbone(test_input)

print(test_img[:, :, 0])
print(test_output[0, 0])
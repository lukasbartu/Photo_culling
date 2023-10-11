__author__ = 'Lukáš Bartůněk'

import os
import json
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from PIL import Image

class NIMA(nn.Module):

    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        out_f = self.features(x)
        out = out_f.view(out_f.size(0), -1)
        out = self.classifier(out)
        return out_f,out

def prepare_model(model_pth):
    base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model = NIMA(base_model)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), model_pth), map_location=torch.device('cpu')))
    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device


# TODO - separate technical and aesthetics quality assessment
def calculate_qualities(pth, lst, result_pth, model_pth):
    if os.path.exists(result_pth):
        return
    model, device = prepare_model(model_pth)
    q_list = [] # list to store results
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mean, std = 0.0, 0.0
    for i ,img in enumerate(lst):
        im = Image.open(os.path.join(pth, str(img))).convert('RGB')
        imt = test_transform(im)
        imt = imt.unsqueeze(dim=0)
        imt = imt.to(device)
        with torch.no_grad():
            out_f, out_class = model(imt)
        out_class = out_class.view(10, 1)
        for j, e in enumerate(out_class, 1):
            mean += j * e
        for k, e in enumerate(out_class, 1):
            std += e * (k - mean) ** 2
        std = std ** 0.5
        q_list += [{"id": i,
                    "img": lst[i],
                    "quality_mean": float(mean),
                    "quality_std": float(std)}]
        mean, std = 0.0, 0.0
    with open(os.path.join(os.getcwd(), result_pth), "w") as write_file:
        json.dump(q_list, write_file, indent=2)
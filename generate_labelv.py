import numpy as np
import torch
import h5py
import torchvision.models as models
import torch.nn.functional as F
from tqdm import tqdm 

if __name__ == "__main__":
    labels = []
    # model = resnet()
    model = models.resnet18(pretrained=True) #torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.cuda()
    model.eval()
    aggre = torch.from_numpy(np.load('utils/cluster_v3.npy')).cuda().float()
    print(aggre.size())
    # print(aggre, aggre.sum(0), aggre.sum(1))
    images = h5py.File('data/Video.h5', 'r')['video']
    with torch.no_grad():
        for im in tqdm(images):
            im = (im/255.0-np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            im = torch.FloatTensor(im).unsqueeze(0)
            im = im.permute(0, 3, 1, 2)
            
            prob = F.softmax(model(im.cuda()), dim=1)
            # print(prob.size(), torch.topk(prob, dim=1, k=990))
            # prob[torch.topk(prob, dim=1, k=990)] = 0
            prediction = torch.matmul(prob, aggre)
            # prediction = torch.sum(prediction, 1)
            # prediction = prediction / torch.max(prediction)
            # print(prediction.squeeze().cpu().numpy().shape)
            labels.append(prediction.squeeze().cpu().numpy())
    np.save('labels_v', labels)

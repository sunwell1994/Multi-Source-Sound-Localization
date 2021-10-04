import numpy as np
import torch
import extract_feature
from models import mobilecrnn_v2
import h5py
from tqdm import tqdm 

if __name__ == "__main__":
    labels = []
    model = mobilecrnn_v2().to('cuda').eval()
    aggre = np.load('utils/cluster_a.npy')
    audios = h5py.File('data/Spec.h5', 'r')['audio']
    with torch.no_grad():
        for feature in tqdm(audios):
            feature = torch.as_tensor(feature).unsqueeze(0).cuda()#.unsqueeze(0)
            # print(feature.size())
            prediction_tag, _ = model(feature)
            prediction = prediction_tag.cpu().detach().numpy() * aggre
            prediction = np.max(prediction, 1)
            #to make the predicted probability more discriminate
            prediction[prediction>0.3] =  prediction[prediction>0.3]*0.4 + 0.6
            labels.append(prediction)
            print(prediction.shape)
            break
    # np.save('labels_a', labels)
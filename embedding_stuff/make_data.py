import numpy as np
import glob

data_paths = glob.glob('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/*OOD.npy')
data_paths.sort()

inp = np.load(data_paths[0], allow_pickle=True)[()]

features = inp['features']
labels = inp['gt_classes'] 
features = features[labels!=3]
labels = labels[labels!=3]
features = features[labels!=4]
labels = labels[labels!=4]
labels += 5


data_paths.pop(0)
# data_paths = data_paths[:500]

for i, path in enumerate(data_paths):
	print(f"index is {i}")
	inp = np.load(path, allow_pickle=True)[()]
	fea = inp['features']
	lab = inp['gt_classes']
	fea = fea[lab!=3]
	lab = lab[lab!=3]
	fea = fea[lab!=4]
	lab = lab[lab!=4]
	lab += 5
	features = np.concatenate((features, fea),axis=0)
	labels = np.concatenate((labels, lab),axis=0)

final_dict = {"features":features, "labels": labels}

np.save("/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data_OOD.npy", final_dict)

import ipdb; ipdb.set_trace()
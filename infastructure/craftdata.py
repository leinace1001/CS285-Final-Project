import numpy as np
import torch
from torch.utils.data import Dataset
import os
import infastructure.pytorch_util as ptu

class CraftData(Dataset):
    def __init__(self, root):
        
        self.data={}
        self.data["image"]=[]
        self.data["action"]=[]
        self.root=root
        for i,file in enumerate(os.listdir(root)):
            tmp=np.load(os.path.join(root,file))
            self.data["image"].append(tmp["image"])
            self.data["action"].append(tmp["action"])
            
            #print(tmp["image"].shape)
        #print(self.data["image"])
        self.data["image"]=np.concatenate(self.data["image"],axis=0)
        self.data["action"]=np.concatenate(self.data["action"],axis=0)
        self.data["image"]=ptu.from_numpy(self.data["image"]).float()
        print(self.data["image"].shape)
        self.data["action"]=ptu.from_numpy(self.data["action"])
        
        
        
    def __getitem__(self, index):
        return self.data["image"][index],self.data["action"][index]
    
    def __len__(self):
        return self.data["image"].shape[0]
        
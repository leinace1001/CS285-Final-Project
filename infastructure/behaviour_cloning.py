import sys
sys.path.append(".")

import pytorch_util as ptu
from policies.MLP_policy import MLPPolicyPG
import gym
import crafter
from craftdata import CraftData
from torch.utils.data import DataLoader
import torch

ptu.init_gpu()
traindata=CraftData("./dataset")
trainLoader=DataLoader(traindata,batch_size=200)
policy=MLPPolicyPG(17, 2, 512,discrete=True)
policy.load_state_dict(torch.load("./output/policy_behavior.pth"))

for epoch in range(1000):
    for i,data in enumerate(trainLoader):
        images,actions=data
        log=policy.update(images, actions)
        if i%100==0:
            print(log)
    print("save weight")
    torch.save(policy.state_dict(),"./output/policy_behavior.pth")
import argparse

import sys
sys.path.append(".")

import crafter
import tqdm
from policies.MLP_policy import MLPPolicyPG
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_noreward-random/0')
parser.add_argument('--steps', type=float, default=1e6)
args = parser.parse_args()

env = crafter.Env()
env = crafter.Recorder(
    env, args.outdir,
    save_stats=True,
    save_episode=False,
    save_video=True,
)
action_space = env.action_space
policy=MLPPolicyPG(17, 2, 512,discrete=True)
policy.load_state_dict(torch.load("./output/policy_behavior.pth"))

done = True
step = 0
bar = tqdm.tqdm(total=args.steps, smoothing=0)
obs = env.reset()
reward=0
while step < args.steps or not done:
  if done:
    reward=0
    obs= env.reset()
    done=False
  action=policy.get_action(obs[None])
  obs, rd, done, info = env.step(action[0])
  reward+=rd
  step += 1
  bar.update(1)
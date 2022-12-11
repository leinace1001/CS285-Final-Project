import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from policies.base_policy import BasePolicy
from torch.nn.modules.loss import _Loss
import infastructure.pytorch_util as ptu


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-5,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        self.observation_net=nn.Sequential(
          nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
          nn.ReLU(),
          nn.Flatten()
        )
        self.observation_net.to(ptu.device)

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=1024,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=1024,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=1024,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from HW1
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
      
        observation=ptu.from_numpy(observation)
        
        tmp=self.forward(observation)
        
        tmp=tmp.sample()
        
        
        
        return ptu.to_numpy(tmp)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, images: torch.FloatTensor):
        images[:,21:29,28:36,:]=0
        images=images.permute(0,3,1,2)
        observation=self.observation_net(images)
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution

#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages=None, q_values=None):
        

        distri=self.forward(observations)
        tmp=distri.log_prob(actions)
        
        self.optimizer.zero_grad()
        
        loss_J=-tmp.mean()
        loss_J.backward()
        self.optimizer.step()
        
        train_log = {
            'loss_J':loss_J.item()

        }
        if self.nn_baseline:
            ## TODO: update the neural network baseline using the q_values as
            ## targets. The q_values should first be normalized to have a mean
            ## of zero and a standard deviation of one.

            ## Note: You will need to convert the targets into a tensor using
                ## ptu.from_numpy before using it in the loss
            torch.cuda.empty_cache()
            q_values=utils.normalize(q_values, np.mean(q_values), np.std(q_values))
            q_values=ptu.from_numpy(q_values)
            self.baseline_optimizer.zero_grad()
            y=self.baseline(observations).view(-1)
            loss=self.baseline_loss(y,q_values)
            loss.backward()
            self.baseline_optimizer.step()
            train_log["Baseline Loss"]=ptu.to_numpy(loss)

      

        
        return train_log

    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())

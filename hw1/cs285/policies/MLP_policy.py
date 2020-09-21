import numpy as np
import torch
import torch.nn as nn
from .base_policy import BasePolicy

class MLPPolicy(nn.Module):

    def __init__(self,
        ac_dim, # output size
        ob_dim, # input size
        n_layers, 
        size, # hidden size
        device,
        learning_rate=1e-4,
        training=True,
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        
        self.training = training
        self.device = device

        self.logstd = nn.Parameter(torch.Tensor([0])) # gaussian standard deviation
        # network architecture
        # TODO -build the network here
        # HINT -build an nn.Modulelist() using the passed the parameters
        self.module_list=nn.ModuleList()

        self.module_list.append(nn.Linear(ob_dim, size))
        self.module_list.append(nn.Tanh())

        for _ in range(n_layers - 1):
            # HINT: use torch.nn.Linear() + torch.nn.Tanh()
            self.module_list.append(nn.Linear(size, size))
            self.module_list.append(nn.Tanh())
        
        self.module_list.append(nn.Linear(size, ac_dim))
        
        if self.training:
            self.loss_func = nn.MSELoss(reduction='sum')
            self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

        self.to(device)

    ##################################

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x


    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def restore(self, filepath):
        self.model.load_state_dict(torch.load(filepath))

    ##################################

    # 提取

    # def action_sampling(self, observation):
    #     observation = torch.tensor(observation)
    #     mean=self.model(observation)
    #     self.sample_ac = mean + torch.exp(self.logstd) * torch.normal(0, 1, mean.size())
    #     return self.sample_ac
    # # 执行训练操作
    # def define_train_op(self):
    #     raise NotImplementedError

    # query this policy with observation(s) to get selected action(s)
    def _get_action(self, obs):

        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        mean = self(torch.Tensor(observation).to(self.device))
        action = mean + torch.exp(self.logstd) * torch.normal(0, 1, mean.size())

        return action

    def get_action(self, obs):
        return self._get_action(obs).cpu().detach().numpy()

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):

    """
        This class is a special case of MLPPolicy,
        which is trained using supervised learning.
        The relevant functions to define are included below.
    """

    # def define_placeholders(self):
    #     # placeholder for observations
    #     self.observations_pl = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)

    #     # placeholder for actions
    #     self.actions_pl = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)

    #     if self.training:
    #         self.acs_labels_na = tf.placeholder(shape=[None, self.ac_dim], name="labels", dtype=tf.float32)

    # def define_train_op(self):
    #     true_actions = self.acs_labels_na
    #     predicted_actions = self.sample_ac

    #     # TODO define the loss that will be used to train this policy
    #     # HINT1: remember that we are doing supervised learning
    #     # HINT2: use tf.losses.mean_squared_error
    #     self.loss = tf.losses.mean_squared_error(true_actions, predicted_actions)
        

    def update(self, observations, actions):
        assert(self.training, 'Policy must be created with training=True in order to perform training updates...')
        
        # Bugfix: convergence is too slow with 1000 step, this is a test remedy that works so well.
        # self.learning_rate=self.learning_rate*0.995

        self.optimizer.zero_grad()
        predicted_actions = self._get_action(observations)
        loss = self.loss_func(predicted_actions, torch.Tensor(actions).to(self.device))
        loss.backward()
        print("Loss = {}".format(loss)) # added by fangda 
        self.optimizer.step()      

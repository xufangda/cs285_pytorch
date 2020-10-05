import numpy as np
import torch
import torch.nn as nn
from .base_policy import BasePolicy

class MLP(BasePolicy):
    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        device,
        learning_rate=1e-4,
        training=True,
        policy_scope='policy_vars',
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.device = device

        if not self.discrete:
            self.logstd = nn.Parameter(torch.zeros(ac_dim)) # gaussian standard deviation
        # network architecture
        # TODO -build the network here
        # HINT -build an nn.Modulelist() using the passed the parameters

        self.module_list=nn.ModuleList()

        self.module_list.append(nn.Linear(ob_dim, size))
        self.module_list.append(nn.Tanh())

        for _ in range(n_layers - 1):
            self.module_list.append(nn.Linear(size, size))
            self.module_list.append(nn.Tanh())
        
        self.module_list.append(nn.Linear(size, ac_dim))
        
        if self.training:
            self.loss_func = nn.MSELoss(reduction='sum')
            self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

        self.to(device)


    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        
        if self.discrete:
            return x
        else:
            return (x, self.logstd.exp())
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def restore(self, filepath):
        self.model.load_state_dict(torch.load(filepath))

    ##################################
    
    
class MLPPolicy:

    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        device,
        learning_rate=1e-4,
        training=True,
        policy_scope='policy_vars',
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.device = device
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        

        self.policy_mlp = MLP(ac_dim, ob_dim, n_layers, size, device,discrete=discrete)
        params = list(self.policy_mlp.parameters())
        if self.nn_baseline:
            self.baseline_mlp = MLP(1, ob_dim, n_layers, size, device, discrete=True)
            params += list(self.baseline_mlp.parameters())
        
        if self.training:
            self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    def update(self, observations, actions):
        raise NotImplementedError


    def get_action(self, obs):
        output = self.policy_mlp(torch.Tensor(obs).to(self.device))
        if self.discrete:
            action_probs = nn.functional.log_softmax(output).exp()
            return torch.multinomial(action_probs, num_samples = 1).cpu().detach().numpy()
        else:
            return torch.normal(output[0], output[1]).cpu().detach().numpy()

    def get_log_prob(self, network_outputs, actions_taken):
        actions_taken = torch.Tensor(actions_taken).to(self.device)
        if self.discrete:
            #log probability under a categorical distribution
            network_outputs = nn.functional.log_softmax(network_outputs).exp()
            return torch.distributions.Categorical(network_outputs).log_prob(actions_taken)
        else:
            #log probability under a multivariate gaussian
            return torch.distributions.Normal(network_outputs[0],network_outputs[1]).log_prob(actions_taken)
   
#####################################################

class MLPPolicyPG(MLPPolicy):

    def update(self, observations, acs_na, adv_n=None, acs_labels_na=None, qvals=None):
        """
        observations - observations
        acs_na - actions_n
        adv_n - advantage_n
        acs_labels_na - actions labels n
        qvals - qvalues 
        """
        policy_output=self.policy_mlp(torch.Tensor(observations).to(self.device))
        logprob_pi = self.get_log_prob(policy_output, acs_na)

        self.optimizer.zero_grad()

        # option 1 loss: mutiply by element
        loss = torch.sum((-logprob_pi * torch.Tensor(adv_n).to(self.device)))

        # option 2 loss: mutiply by sum
        # loss = -torch.sum(logprob_pi)*torch.sum(torch.Tensor(adv_n).to(self.device))

        loss.backward()

        if self.nn_baseline:
            baseline_prediction = self.baseline_mlp(torch.Tensor(observations).to(self.device)).view(-1)
            baseline_target = torch.Tensor(qvals - qvals.mean())/(qvals.std() + 1e-8).to(self.device)
            baseline_loss = nn.functional.mse_loss(baseline_prediction, baseline_target)
            baseline_loss.backward()

        self.optimizer.step()

        return loss



import numpy as np
import torch
import torch.nn as nn
from .base_policy import BasePolicy


class MLPPolicy(BasePolicy):

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


    #  def define_forward_pass(self):
    #     if self.discrete:
    #         logits_na = build_mlp(self.observations_pl, output_size=self.ac_dim, scope='discrete_logits', n_layers=self.n_layers, size=self.size)
    #         self.parameters = logits_na
    #     else:
    #         mean = build_mlp(self.observations_pl, output_size=self.ac_dim, scope='continuous_logits', n_layers=self.n_layers, size=self.size)
    #         logstd = tf.Variable(tf.zeros(self.ac_dim), name='logstd')
    #         self.parameters = (mean, logstd)

    # def build_action_sampling(self):
    #     if self.discrete:
    #         logits_na = self.parameters
    #         self.sample_ac = tf.squeeze(tf.multinomial(logits_na, num_samples=1), axis=1)
    #     else:
    #         mean, logstd = self.parameters
    #         self.sample_ac = mean + tf.exp(logstd) * tf.random_normal(tf.shape(mean), 0, 1)

    def define_log_prob(self):
        if self.discrete:
            #log probability under a categorical distribution
            logits_na = self.parameters
            self.logprob_n = tf.distributions.Categorical(logits=logits_na).log_prob(self.actions_pl)
        else:
            #log probability under a multivariate gaussian
            mean, logstd = self.parameters
            self.logprob_n = tfp.distributions.MultivariateNormalDiag(
                loc=mean, scale_diag=tf.exp(logstd)).log_prob(self.actions_pl)

    def build_baseline_forward_pass(self):
        self.baseline_prediction = tf.squeeze(build_mlp(self.observations_pl, output_size=1, scope='nn_baseline', n_layers=self.n_layers, size=self.size))

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def restore(self, filepath):
        self.model.load_state_dict(torch.load(filepath))

    ##################################
    
    # query the neural net that's our 'policy' function, as defined by an mlp above
    # query the policy with observation(s) to get selected action(s)
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

    def define_train_op(self):
        raise NotImplementedError

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

#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):

    # def define_placeholders(self):
    #     # placeholder for observations
    #     self.observations_pl = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)

    #     # placeholder for actions
    #     if self.discrete:
    #         self.actions_pl = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
    #     else:
    #         self.actions_pl = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)

    #     if self.training:
    #         # placeholder for advantage
    #         self.adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

    #         if self.nn_baseline:
    #             # targets for baseline
    #             self.targets_n = tf.placeholder(shape=[None], name="baseline_target", dtype=tf.float32)

    #########################

    def define_train_op(self):

        # define the log probability of seen actions/observations under the current policy
        self.define_log_prob()

        # TODO: define the loss that should be optimized when training a policy with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: see define_log_prob (above)
            # to get log pi(a_t|s_t)
        # HINT3: look for a placeholder above that will be populated with advantage values 
            # to get [Q_t - b_t]
        # HINT4: don't forget that we need to MINIMIZE this self.loss
            # but the equation above is something that should be maximized
        self.loss = tf.reduce_sum(TODO)

        # TODO: define what exactly the optimizer should minimize when updating the policy
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(TODO)

        if self.nn_baseline:
            # TODO: define the loss that should be optimized for training the baseline
            # HINT1: use tf.losses.mean_squared_error, similar to SL loss from hw1
            # HINT2: we want predictions (self.baseline_prediction) to be as close as possible to the labels (self.targets_n)
                # see 'update' function below if you don't understand what's inside self.targets_n
            self.baseline_loss = TODO

            # TODO: define what exactly the optimizer should minimize when updating the baseline
            self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(TODO)

    #########################

    def train_step(self, obs):
        pass

    def run_baseline_prediction(self, obs):
        
        # TODO: query the neural net that's our 'baseline' function, as defined by an mlp above
        # HINT1: query it with observation(s) to get the baseline value(s)
        # HINT2: see build_baseline_forward_pass (above) to see the tensor that we're interested in
        # HINT3: this will be very similar to how you implemented get_action (above)
        return TODO

    def update(self, observations, acs_na, adv_n=None, acs_labels_na=None, qvals=None):
        assert(self.training, 'Policy must be created with training=True in order to perform training updates...')

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.observations_pl: observations, self.actions_pl: acs_na, self.adv_n: adv_n})

        if self.nn_baseline:
            targets_n = (qvals - np.mean(qvals))/(np.std(qvals)+1e-8)
            # TODO: update the nn baseline with the targets_n
            # HINT1: run an op that you built in define_train_op
            TODO
        return loss

#####################################################
#####################################################

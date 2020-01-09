import numpy as np
import tensorflow as tf
from .base_policy import BasePolicy
from cs285.infrastructure.tf_utils import build_mlp
import tensorflow_probability as tfp

class MLPPolicy(BasePolicy):

    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
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
        self.size = size
        self.learning_rate = learning_rate
        self.training = training

        # build TF graph
        self.build_model()
        self.optimizer=tf.keras.optimizers.Adam(self.learning_rate)
        # saver for policy variables that are not related to training
        # self.policy_vars = [v for v in tf.all_variables() if policy_scope in v.name and 'train' not in v.name]
        # self.policy_saver = tf.train.Saver(self.policy_vars, max_to_keep=None)

    ##################################
    
    # TF1建图，没什么用 2020.1.5
    # Change to TF2 Build model 2020.1.6 @Fangda
    def build_model(self):
        # self.define_placeholders()
        model= build_mlp(output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
        self.model=model
        self.logstd = tf.Variable(tf.zeros(self.ac_dim), name='logstd')
        
    ##################################

    def forward_pass(self, observation):
        return self.model(observation)


    @tf.function
    def action_sampling(self,observation):
        mean=self.model(observation)
        self.sample_ac = mean + tf.math.exp(self.logstd) * tf.random.normal(tf.shape(mean), 0, 1)
        return self.sample_ac
    # # 执行训练操作
    # def define_train_op(self):
    #     raise NotImplementedError

    ##################################

    # TF1的checkpoint保存和回复
    def save(self, filepath):
        self.model.save_weights(filepath)
    def restore(self, filepath):
        self.model.load_weights(filepath)
    ##################################

    # 提取

    # query this policy with observation(s) to get selected action(s)
    def get_action(self, obs):

        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        # HINT1: you will need to call self.sess.run
        # HINT2: the tensor we're interested in evaluating is self.sample_ac
        # HINT3: in order to run self.sample_ac, it will need observation fed into the feed_dict
        sample_ac=self.action_sampling(observation)

        return sample_ac

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
        # self.sess.run(self.train_op, feed_dict={self.observations_pl: observations, self.acs_labels_na: actions})
        
        # Bugfix: convergence is to slow with 1000 step, this is a test remedy that works so well.
        # self.learning_rate=self.learning_rate*0.995

       
        # Add TF2 1/4/2020
        with tf.GradientTape() as tape:
            predicted_actions = self.action_sampling(observations) # Forward pass of the model
            loss_n = tf.losses.mean_squared_error(actions, predicted_actions)
            loss=tf.reduce_sum(loss_n)
            # print("Loss = {}".format(loss))
        params=[self.logstd]+ self.model.variables
        grads = tape.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

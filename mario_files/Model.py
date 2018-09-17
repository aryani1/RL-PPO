import tensorflow as tf
import numpy as np
import gym

from baselines.common import explained_variance
from abc import ABC, abstractmethod

# abstractenvrunner from openai runners
class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

class Model(object):
# The model class

    def __init__(self, policy, ob_space, action_space, nenvs, nsteps, ent_coef,
                v_coef, max_grad_norm):

        sess = tf.get_default_session()
        
        # Placeholders for variables in the loss function
        actions_    = tf.placeholder(tf.int32, [None], name='actions_')
        
        advantages_ = tf.placeholder(tf.float32, [None], name='advantages_')
        rewards_    = tf.placeholder(tf.float32, [None], name='rewards_')
        lr_         = tf.placeholder(tf.float32, name='learning_rate_')

        # old actor
        oldneglopac_ = tf.placeholder(tf.float32, [None], name='oldneglopac_')
        # old critic
        oldvpred_    = tf.placeholder(tf.float32, [None], name='oldvpred_')
        # clip range
        cliprange_   = tf.placeholder(tf.float32, [])

        '''
        Step model is used for sampling experience by the different environments
        Train model is used to average the experience and update gradients. (A2C)
        '''
        step_model  = policy(sess, ob_space, action_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, action_space, nenvs*nsteps, nsteps, reuse=True)

        # Calculate the total loss
        
        # first calculate the loss for the state value function
        # get the predicted state value and clip it
        v_pred         = train_model.v 
        v_pred_clipped = oldvpred_ + tf.clip_by_value(train_model.v - oldvpred_,
                                                      -cliprange_,
                                                      cliprange_)

        v_loss         = tf.square(v_pred - rewards_)
        v_loss_clipped = tf.square(v_pred_clipped - rewards_)

        value_loss     = 0.5 * tf.reduce_mean(tf.maximum(v_loss, v_loss_clipped))
        print(train_model.pi)
        print(actions_)
        # calculate the loss for the policy
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=actions_)
        ratio = tf.exp(oldneglopac_ - neglogpac)

        pg_loss         = -advantages_ * ratio
        pg_loss_clipped = -advantages_ * tf.clip_by_value(ratio, 1.0 - cliprange_, 1.0+cliprange_)

        policy_gradient_loss = tf.reduce_mean(tf.maximum(pg_loss, pg_loss_clipped))

        # add entropy to support exploration for our policy
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # define the total loss
        loss = policy_gradient_loss - entropy * ent_coef + value_loss * v_coef
        print('LOLOLOL')
        # calculate and apply gradients
        optimizer            = tf.train.AdamOptimizer(learning_rate=lr_)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _         = tf.clip_by_global_norm(gradients, max_grad_norm)
        minimize             = optimizer.apply_gradients(zip(gradients, variables))

        def train(s, a, r, values, neglogpacs, lr, cliprange):
            # calculate the advantages
            advantages = r - values

            # normalize the advantages (?)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            data_dict = {train_model.inputs_: s,
                         actions_ : a,
                         advantages_ : advantages,
                         rewards_ : r, 
                         lr_: lr,
                         cliprange_ : cliprange,
                         oldneglopac_ : neglogpacs,
                         oldvpred_ : values}

            p_l, v_l, p_e, _ = sess.run([policy_gradient_loss, value_loss, entropy, minimize], data_dict)

            return p_l, v_l, p_e

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        # self.save = save
        # self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, total_timesteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)

        # discount factor
        self.gamma = gamma

        # lambda for generalized advantage estimation
        self.lam = lam

        self.total_timesteps = total_timesteps
    
    def run(self):
        mb_obs, mb_actions, mb_rewards, mb_values, mb_neglopacs, mb_dones = [],[],[],[],[],[]

        for _ in range(self.nsteps):
            actions, values, neglopacs = self.model.step(self.obs, self.dones)

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglopacs.append(neglopacs)
            mb_dones.append(self.dones)

            print(actions)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            mb_rewards.append(rewards)

        mb_obs       = np.asarray(mb_obs, dtype=np.uint8)
        mb_actions   = np.asarray(mb_actions, dtype=np.int32)
        mb_rewards   = np.asarray(mb_rewards, dtype=np.float32)
        mb_values    = np.asarray(mb_values, dtype=np.float32)
        mb_neglopacs = np.asarray(mb_neglopacs, dtype=np.float32)
        mb_dones     = np.asarray(mb_dones, np.bool)
        last_values  = self.model.value(self.obs)

        ## Generalized advantage estimation (GAE)
        mb_returns    = np.zeros_like(mb_rewards)
        mb_advantages = np.zeros_like(mb_rewards)

        # last lambda from GAE
        last_lam = 0

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            d = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advantages[t] = last_lam = d + self.gamma * self.lam * nextnonterminal * last_lam
        mb_returns = mb_advantages + mb_values
        return map(sf01, (mb_obs, mb_actions, mb_returns, mb_values, mb_neglopacs))
            
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def learn(policy, env, nsteps, total_timesteps, gamma, lam, v_coef, ent_coef, lr,
          cliprange, max_grad_norm, log_interval):

        noptepochs = 4
        nminibatches = 8
        
        # number of envs
        nenvs = env.num_envs
    
        # observation and action space
        ob_space = env.observation_space

        # try out different action space
        ac_space = env.action_space
        #ac_space = gym.spaces.Discrete(6)

        # set the batch size
        batch_size = nenvs * nsteps
        batch_train_size = batch_size // nminibatches

        assert batch_size % nminibatches == 0

        model = Model(policy=policy,
                        ob_space=ob_space,
                        action_space=ac_space,
                        nenvs=nenvs,
                        nsteps=nsteps,
                        ent_coef=ent_coef,
                        v_coef=v_coef,
                        max_grad_norm=max_grad_norm)
        
        runner = Runner(env, model, nsteps=nsteps, total_timesteps=total_timesteps, gamma=gamma, lam=lam)
        nupdates = total_timesteps//batch_size+1

        for update in range(1, nupdates):

            frac = 1.0 - (update - 1.0) / nupdates
            learning_rate = frac
            clip_range    = frac

            # get the minibatch
            obs, actions, returns, values, neglogpacs = runner.run()

            mb_losses = []
            total_batches_train = 0

            indices = np.arange(batch_size)

            for _ in range(noptepochs):
                np.random.shuffle(indices)

                for batch_start in range(0, batch_size, batch_train_size):
                    batch_end = batch_start + batch_train_size
                    mbinds    = indices[batch_start:batch_end]
                    slices    = (arr[mbinds] for arr in (obs, actions, returns, values, neglogpacs))
                    mb_losses.append(model.train(*slices, learning_rate, clip_range))

def play(policy, env, update):
    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(policy=policy,
                  ob_space=ob_space,
                  action_space=ac_space,
                  nenvs=1,
                  nsteps=1,
                  ent_coef=0,
                  v_coef=0,
                  max_grad_norm=0)
    obs = env.reset()

    score = 0
    done = False
    
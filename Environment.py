import gym
import multiprocessing
from   multiprocessing import Process, Lock

from Agent import Agent
from Network import Network
import pickle
import numpy   as np

##############
#### REMEMBER: 
#### If errors with synchronization and update, then check that the queues are of the right type
##############

class Environment:
    '''
    This is the Environment class, it is responsible for 
    spawning a new thread and using an agent to act and train
    '''

    def __init__(self, n_step, gamma, queue_sync, queue_upd, n_processors, g_counter, env, isGlobal=False): # removed network param
        from   Network import Network
        from   Agent   import Agent
        import keras

        self.n_step       = n_step
        self.gamma        = gamma
        self.isGlobal     = isGlobal
        self.queue_sync   = queue_sync
        self.queue_upd    = queue_upd
        self.n_processors = n_processors


        # global and local counter
        self.g_counter = g_counter
        self.counter = 0
        self.lock    = Lock()

        # create new random seed for each child process
        np.random.seed()

        if isGlobal:
            self.network = Network(self.gamma, self.n_step, self.queue_sync, self.n_processors, self.isGlobal)
            self.run_sync_agents()
        else:
            self.run(env)
    
    # Synchronize the agents as long as they are training. This function is mainly
    # used by the global network.
    def run_sync_agents(self):
        while True:
            # start synchronization with the agents every 'x' timestep
            if self.g_counter.value % 500 == 0:
                while not self.queue_sync.empty():
                    _ = self.queue_sync.get()

                for _ in range(self.n_processors):
                    # send the global network's weights to the agents
                    weights = self.network.get_weights()
                    self.queue_sync.put(self.pickle_weights(weights))
                self.increment_global_counter()
                print('GLOBAL NET: Syncing weights to agents!')
                self.network.model.save('mario_model.h5')

            # update the global network's weights when there is d_w in the queue
            while not self.queue_upd.empty():
                # d_w is the change of the weights
                d_w = self.unpickle_weights(self.queue_upd.get())
                self.network.update_weights(d_w)
                print('GLOBAL NET: Updated weights from an agent!')

            # self.increment_global_counter()

    # initialize the network, environment and the agent and
    # then run the training.
    def run(self, env):
        self.network = Network(self.gamma, self.n_step, self.queue_sync, self.n_processors, self.isGlobal)
        self.env     = env
        self.agent   = Agent(self.env, self.n_step, self.gamma, self.network)
        self.env.reset()

        self.run_episode()

    def run_episode(self):
        # Reset the env
        s = self.env.reset()
        self.init_env()

        # Reset the agent
        self.agent.init_frames()

        # Act and observe until were done
        while True:
            action_idx, action = self.agent.act()
            s_prim, reward, done, info = self.env.step(action)

            # process the rewards
            done, reward_processed = self.process_reward(reward, info, done)

            if done:
                s_prim = None

            onehot_action = np.zeros(len(self.agent.actions))
            onehot_action[action_idx] = 1
            has_updated = self.agent.train(s, onehot_action, reward_processed, s_prim)
            #s = s_prim

            if done: # or self.stop_signal
                self.init_env()
                self.agent.init_frames()
                self.env.change_level(0)
                continue
            else:
                s = s_prim
                self.agent.next_frame(s)

            # increment the local and global counter after each episode
            self.increment_local_counter()
            self.increment_global_counter()

            if has_updated:
                w = self.network.get_weights()
                delta_w = w - self.network.get_global_weights()
                self.queue_upd.put(self.pickle_weights(delta_w))

            if self.counter % 100 == 0:
                if not self.queue_sync.empty():
                    new_w = self.unpickle_weights(self.queue_sync.get())
                    self.network.set_weights(new_w)
                    self.network.set_global_weights(np.array(new_w))
                    self.counter = 0 # prevent unecessarily large numbers

    def init_env(self):
        self.gameInfo = {'max_distance': 0, 'time':400, 'score':0, 'staleness':0}
        #s = self.env.reset()

    def get_state_dim(self):
        return (self.env.observation_space.shape[0], self.env.observation_space.shape[1])

    # increments the local counter
    def increment_local_counter(self):
        self.counter += 1

    # Increment the global counter
    def increment_global_counter(self):
        with self.g_counter.get_lock():
            self.g_counter.value = self.g_counter.value + 1

    # pickle the weights so they can be passed through a queue
    def pickle_weights(self, w):
        # use protocol=-1 to use the latest protocol
        return pickle.dumps(w, protocol=-1)

    def unpickle_weights(self, w):
        return pickle.loads(np.array(w))

    def process_reward(self, reward, info, done):
        ##### TODO: FIX REWARD WHEN RESPAWNING
        r = 0
        if 'distance' in info:

            if info['distance'] > self.gameInfo['max_distance']:
                self.gameInfo['max_distance'] = info['distance']
                self.gameInfo['staleness']    = 0
            else:
                self.gameInfo['staleness']   += 1

            r += reward * 0.5

            # Check time
            if info['time'] < self.gameInfo['time']:
                r -= 0.01

            # Check score
            r += (info['score'] - self.gameInfo['score']) * 0.0001 # tune

            if info['life'] == 0 or self.gameInfo['staleness'] > 200: # tune staleness
                r -= 1
                done = True

            if done and info['distance'] > 0.97 * 3266: # 3266 is max_distance @ level 1
                r += 1

            self.gameInfo['time']     = info['time']
            self.gameInfo['score']    = info['score']
            
            return done, r
        return done, r


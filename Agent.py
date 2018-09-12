import numpy as np

class Agent:
    def __init__(self, env, n_step, gamma, network):
        import keras
        import tensorflow
        self.n_step    = n_step
        self.gamma     = gamma
        self.network   = network
        self.n_actions = 12
        self.memory    = []
        self.timelapse = []

        self.env = env
        self.print_counter = 0

        self.actions = [
           [0, 0, 0, 0, 0, 0], #0 - no button",
           [1, 0, 0, 0, 0, 0], #1 - up only (to climb vine)",
           [0, 0, 1, 0, 0, 0], #2 - left only",
           [0, 0, 0, 1, 0, 0], #3 - right only",
           [0, 0, 0, 0, 0, 1], #4 - run only",
           [0, 0, 0, 0, 1, 0], #5 - jump only",
           [0, 0, 1, 0, 0, 1], #6 - left run",
           [0, 0, 1, 0, 1, 0], #7 - left jump",
           [0, 0, 0, 1, 0, 1], #8 - right run",
           [0, 0, 0, 1, 1, 0], #9 - right jump",
           [0, 0, 1, 0, 1, 1], #10 - left run jump",
           [0, 0, 0, 1, 1, 1]] #11 - right run jump",

    def act(self):   
        '''
        Choose an action based on a policy p
        '''

        a = np.zeros(self.n_actions, dtype=np.int)

        p = self.network.predict_p(self.timelapse)[0]
        
        self.print_counter += 1
        if self.print_counter % 1000 == 0:
            print(p)
            self.print_counter = 0

        # Pick the action from the probabilities given
        # by the policy p
        action_num = np.random.choice(self.n_actions, p=p)

        return action_num, self.actions[action_num]

    # train da mario
    def train(self, s, a, r, s_prim):
        updated_network = False
        self.add_to_memory(s, a, r, s_prim)

        if self.has_n_step() or self.is_done(s_prim):
            s, a, r, s_ = self.get_n_step()
            self.network.train_push(s, a, r, s_)
            self.memory = []
            updated_network = True
            
        return updated_network

    # TODO: s is not really used
    def add_to_memory(self, s, a, r, s_prim):
        '''
            Add the observations from different states to memory
        '''
        s = self.timelapse

        if s_prim is not None:
            # copy the timelapse values
            next_timelapse = self.timelapse.copy()

            # change the last frame
            next_timelapse[:-1] = next_timelapse[1:]
            next_timelapse[-1]  = s_prim
            s_prim = next_timelapse

        # store actions and observations in memory
        self.memory.append( (s, a, r, s_prim) )

    def is_done(self, state):
        if state is None:
            return True
        else:
            return False

    # return the n-step return from the memory
    def get_n_step(self):
        mem_len = min(self.n_step, len(self.memory))
        state, action, _, _ = self.memory[0]
        _, _, _, last_state = self.memory[mem_len-1]

        # test inefficient method
        r = 0
        for i in range(mem_len):
            r += self.memory[i][2] * (self.gamma ** i)

        return state, action, r, last_state

    def has_n_step(self):
        if len(self.memory) >= self.n_step:
            return True
        else:
            return False

    def init_frames(self):
        # constants to change if one uses pixels/blocks, and
        # if you want to change number of images used for
        # predicting.
        frames_x = 13
        frames_y = 16
        n_imgs = 4

        self.timelapse = np.zeros((n_imgs, frames_x, frames_y))

    def next_frame(self, s):
        self.timelapse[:-1] = self.timelapse[1:]
        self.timelapse[-1]  = s


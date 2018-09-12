import numpy as np
import threading
import pickle
from keras.layers import TimeDistributed

class Network:
    '''
    The network class is responsible for the neural network
    and the training of it using a synchronized buffer
    holding training samples.
    '''

    def __init__(self, gamma, n_step, queue, n_processors, isGlobal=False):
        import tensorflow as tf
        import keras
        from keras import backend as K

        self.img_shape_x = 13
        self.img_shape_y = 16
        self.lock_queue = threading.Lock()
        self.batch_size = 32
        self.num_actions = 12 # check agent self.actions 
        self.gamma  = gamma
        self.n_step = n_step
        self.isGlobal = isGlobal
        self.n_processors = n_processors

        self.train_queue  = [[],[],[],[],[]]
        self.n_imgs       = 4

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        # building model and graph
        self.model = self.build_net()
        self.graph = self.build_graph(self.model)

        # initialize graph
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        # Global network should send its initial weights so all workers
        # start with the same weights.
        if isGlobal:
            weights = pickle.dumps(self.model.get_weights(), protocol=-1)
            for _ in range(self.n_processors):
                queue.put(weights)
        else:
            while queue.empty():
                pass
            self.global_net_weights = np.array(pickle.loads(queue.get()))
            self.set_weights(self.global_net_weights)

        # prevent modifications
        #self.default_graph.finalize()

    def build_graph(self, model):
        import tensorflow as tf
        # parameters may need tuning
        loss_v = .5
        loss_e = .01
        rms_decay = .99
        eps = 1e-10 # just a small number
        eta = 1e-4

        s_t = tf.placeholder(tf.float32, shape=(None, self.n_imgs, self.img_shape_x, self.img_shape_y, 1))
        a_t = tf.placeholder(tf.float32, shape=(None, self.num_actions))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + eps)
        advantage = r_t - v

        loss_value = loss_v * tf.square(advantage)
        loss_policy = - log_prob * tf.stop_gradient(advantage)
        entropy = loss_e * tf.reduce_sum(p * tf.log(p + eps), axis=1, keep_dims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        #optimizer = tf.train.RMSPropOptimizer(eta, decay=rms_decay)
        optimizer = tf.train.AdamOptimizer(learning_rate=eta)

        #minimize = optimizer.minimize(loss_total)
        # --- instead of minimize, calculate and clip gradients
        gradients, variables = zip(*optimizer.compute_gradients(loss_total))
        gradients, _ = tf.clip_by_global_norm(gradients, 40.0)
        minimize = optimizer.apply_gradients(zip(gradients, variables))

        return s_t, a_t, r_t, minimize

    def build_net(self):
        import keras
        '''
        CNN working on tiles, input: (13,16,1) img
        connected to FC/Dense of 128 neurons
        connected to two heads one actor and one critic
        '''
        # CNN
        layer_input = keras.layers.Input(shape=(self.n_imgs, self.img_shape_x, self.img_shape_y, 1))
        layer_conv1 = TimeDistributed(keras.layers.Conv2D(16, (2,2), strides=(1,1), padding='valid', activation='relu'))(layer_input)
        layer_conv2 = TimeDistributed(keras.layers.Conv2D(32, (1,1), strides=(1,1), padding='valid', activation='relu'))(layer_conv1)

        layer_flatten = TimeDistributed(keras.layers.Flatten())(layer_conv2)
        layer_lstm  = keras.layers.LSTM(64, activation='tanh')(layer_flatten)

        # hidden'
        layer_hidden = keras.layers.Dense(128, activation='relu')(layer_lstm)

        # outputs
        output_actions = keras.layers.Dense(self.num_actions, activation='softmax')(layer_hidden)
        output_value = keras.layers.Dense(1, activation='linear')(layer_hidden)

        # model
        model = keras.models.Model(inputs=[layer_input], outputs=[output_actions, output_value])
        model._make_predict_function()

        return model

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(np.zeros((self.n_imgs, self.img_shape_x, self.img_shape_y)))
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)
        self.optimizer()

    def optimizer(self):
        '''
        Preprocess and prepare the data for training
        '''
        # if there aren't enough samples in the queue, yield the thread
        if len(self.train_queue[0]) < self.batch_size:
            #time.sleep(0) <--------------------------------------------------- fixa en timer sen
            return None

        # call the __enter__() function on the lock object,
        # this locks the thread until we are done.
        with self.lock_queue:
            if len(self.train_queue[0]) < self.batch_size:
                return None

            state, action, reward, next_state, not_terminal = self.train_queue
            self.train_queue = [[],[],[],[],[]]

        # stack the data vertically in a numpy array
        state = np.vstack(state)
        action = np.vstack(action)
        reward = np.vstack(reward)
        next_state = np.vstack(next_state)
        not_terminal = np.vstack(not_terminal)

        # estimate the state value for next_state
        state_value = self.predict_v(next_state)

        # update the reward by adding the estimated state value for next_state
        # only update if next_state is not a terminal state
        reward += not_terminal * (self.gamma ** self.n_step) * state_value
        
        s_t, a_t, r_t, minimize = self.graph
        
        state_batch = np.reshape(state, (self.batch_size, self.n_imgs, 13, 16, 1))

        self.session.run(minimize, feed_dict={s_t: state_batch, a_t: action, r_t: reward})

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_v(self, s):
        with self.default_graph.as_default():
            _, v = self.model.predict(np.reshape(s, (-1, self.n_imgs, 13, 16, 1)))
            return v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, _ = self.model.predict(np.reshape(s, (-1, self.n_imgs, 13, 16, 1)))
            return p

    # return the network's weights
    def get_weights(self):
        return self.model.get_weights()

    # set the weights of the current network
    def set_weights(self, w):
        self.model.set_weights(w)

    # returns the global network's weights
    def get_global_weights(self):
        return self.global_net_weights

    # sets the global network
    def set_global_weights(self, net):
        self.global_net_weights = net

    # update the weights of the network with d_w (delta weights).
    # only the global network will use this function.
    def update_weights(self, d_w):
        weights = self.model.get_weights()
        for idx, w in enumerate(weights):
            d_w[idx] = w + d_w[idx]

        # new_weights = weights + d_w # Check this again ---------
        self.set_weights(d_w)



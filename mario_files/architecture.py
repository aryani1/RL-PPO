import numpy as np 
import tensorflow as tf

from baselines.common.distributions import make_pdtype

# Helper function for creating convolutional layers
def conv_layer(inputs, filters, kernel_size, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=(strides, strides),
                            kernel_initializer=tf.orthogonal_initializer(gain))

# Helper function for creating fully connected layers
def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=activation_fn,
                           kernel_initializer=tf.orthogonal_initializer(gain))


class PPO(object):
    def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse = False):
        gain = np.sqrt(2) # init kernels
        self.pdtype = make_pdtype(action_space)
        # mario settings - tiles
        x = 13
        y = 16
        channels = 1 # 1 color channel

        #input_shape = (x, y, channels)
        input_shape = (x, y)
        inputs_     = tf.placeholder(tf.float32, [None, *input_shape], name='input')

        with tf.variable_scope("model", reuse = reuse):
            #inp_reshape = tf.reshape(inputs_, shape=(None, x,y,channels))
            inp_reshape = tf.expand_dims(inputs_, 3)
            conv1 = conv_layer(inp_reshape, 16, 2, 1, gain)
            conv2 = conv_layer(conv1, 32, 1, 1, gain)

            flatten = tf.layers.flatten(conv2)

            fc1 = fc_layer(flatten, 128, gain=gain)
            self.pd, self.pi = self.pdtype.pdfromlatent(fc1, init_scale=0.01)
            print(self.pi)
            v = fc_layer(fc1, 1, activation_fn=None)[:, 0]
    
        self.initial_state = None

        action = self.pd.sample()

        # the negative logarithm of our probability
        neglog_action = self.pd.neglogp(action)

        # Function use to take a step returns action to take and V(s)
        def step(state_in, *_args, **_kwargs):
            # return a0, v(s), neglogp0
            return sess.run([action, v, neglog_action], {inputs_: state_in})

        # Function that calculates only the V(s)
        def value(state_in, *_args, **_kwargs):
            return sess.run(v, {inputs_: state_in})

        # Function that output only the action to take
        def select_action(state_in, *_args, **_kwargs):
            return sess.run(action, {inputs_: state_in})

        self.inputs_ = inputs_
        self.v = v
        self.step = step
        self.value = value
        self.select_action = select_action

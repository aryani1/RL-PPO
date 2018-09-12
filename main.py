from Environment import Environment
from Network import Network
import multiprocessing
import pickle
import gym

#### TODO
# Load/Save models         [/]
# Fine tune rewards.       [ ]
# Check reward function.   [x] || using inefficient method atm.
# Add bootstrapped val     [ ] || add bootstrapped values at the end of the episode.
# Look at 'playerstatus'.  [ ]
# Empty memory buffer      [x]
# Update agents more often [x]
# Tensorboard              [ ]
# Action mapping           [x]
# Update agents first!!    [ ]

# Parameters
n_step = 16
gamma = 0.99

n_processors = 1
g_counter = multiprocessing.Value('i', 1)

def main():
    # define the agents and initialize the queue
    # for multiprocess communication.
    agents   = []

    # there are two queues, one for the global synchronization of weights
    # and one for sending updated weights to the global network.
    q_sync   = multiprocessing.Queue()
    q_update = multiprocessing.Queue()

    env = gym.make('SuperMarioBros-1-1-v0')
    env.close()

    # instantiate global network
    g_network = multiprocessing.Process(target=Environment, args=(n_step, gamma, q_sync, q_update, n_processors, g_counter, env, True))

    # create as many workers as n_processors
    for _ in range(n_processors):
        agent = multiprocessing.Process(target=Environment, args=(n_step, gamma, q_sync, q_update, n_processors, g_counter, env))
        agent.daemon = True
        agents.append(agent)

    # start the global network and wait for it to fill the queue with its weights
    g_network.start()

    while q_sync.empty():
        pass

    # start the agents
    for a in agents:
        a.start()

    a.join()

if __name__ == '__main__':
    main()
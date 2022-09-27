from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import _pickle as cPickle


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=True):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, 180, 240, 3))
        self.new_state_memory = np.zeros((self.mem_size, 180, 240, 3))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, s, action, reward, s_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = s
        self.new_state_memory[index] = s_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    """model = Sequential([
        Dense(fc1_dims, input_shape=(input_dims,)),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(n_actions)])
    """
    model = Sequential()
    model.add(Conv2D(64, (7, 7), input_shape=(180, 240, 3, )))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(n_actions, activation='linear'))
    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model


def transform_state(state):
    return state
    #return np.array([(state[0]+1)/43, state[1]/480, state[2]/360])


class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000, fname='ddqn_model', replace_target=10):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname + '.h5'
        self.train_data = fname + '.pkl'
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)


        self.NN_size = 16
        self.q_eval = build_dqn(alpha, n_actions, input_dims, self.NN_size,
                                self.NN_size)
        self.q_target = build_dqn(alpha, n_actions, input_dims, self.NN_size,
                                  self.NN_size)

    def remember(self, state, action, reward, new_state, done):
        if reward > 0:
            self.memory.store_transition(state, action, reward, new_state, done)
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, s, score, max_score):
        s = s[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            print('Random: %.3f < %.3f' % (rand, self.epsilon))
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(s)
            action = np.argmax(actions)
            print('Expect:', actions[0, action])
        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, action_indices] = reward + \
                                                    self.gamma * q_next[
                                                        batch_index, max_actions.astype(
                                                            int)] * done
            self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
                                                              self.epsilon_min else self.initial_epsilon
            if self.memory.mem_cntr % self.replace_target == 0:
                print('\nUpdate Network')
                self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        print('\nSaving Data, Do Not Stop the Program')
        self.q_eval.save(self.model_file)
        print('\nModel Saved ')
        try:
            f = open(self.train_data, 'wb')
            cPickle.dump(self.memory, f, protocol=-1 )
            f.close()
            print('\nSuccessfully Saved Train Data')
        except:
            f.close()
            print('\nFailed to Save Train Data')

    def load_model(self):

        try:
            f = open(self.train_data, 'rb')
            self.memory = cPickle.load(f)
            f.close()
            print('\nSuccessfully Loaded Train Data')
        except:
            print('\nFailed to Load Train Data')
            f.close()

        self.q_eval = load_model(self.model_file)
        self.update_network_parameters()
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()

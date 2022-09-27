from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import _pickle as cPickle

import cv2
class ReplayBuffer(object):
    def __init__(self, max_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.actions = np.zeros(self.mem_size)
        self.rewards = np.zeros(self.mem_size)
        self.PATH = 'C:/Users/ZhaoLiu/Desktop/JKData'

    def store_transition(self, s, action, reward, s_):
        index = self.mem_cntr % self.mem_size
        path1 = self.PATH + '/' + str(index)+'.jpg'
        path2 = self.PATH + '/' + str(index)+'_.jpg'
        cv2.imwrite(path1, s)
        cv2.imwrite(path2, s_)
        cv2.waitKey(0)
        self.actions[index] = action
        self.rewards[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = []
        states_ = []
        for index in batch:
            states.append(cv2.imread(self.PATH+'/'+str(index)+'.jpg'))
            states_.append(cv2.imread(self.PATH+'/'+str(index)+'_.jpg'))
        actions = self.actions[index]
        rewards = self.rewards[index]

        return states, actions, rewards, states_

def build_dqn(lr):
    model = Sequential()
    model.add(Conv2D(128, (5, 5), input_shape=(360, 480, 3), activation='relu'))
    model.add(Conv2D(64, (5, 5), input_shape=(360, 480, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(16))
    model.add(Dense(4))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(lr=lr), loss = 'mse')

    return model

class DDQN(object):
    def __init__(self, alpha, gamma, epsilon, batch_size,
                 epsilon_dec=0.99, epsilon_end=0.01, mem_size=1000,
                 fname='ddqn_model', replace_target=10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname + '.h5'
        self.train_date = fname + '.pkl'
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size)

        self.q_eval = build_dqn(alpha)
        self.q_target = build_dqn(alpha)

    def remember(self, state, action, reward):
        self.memory.store_transition(state, action, reward)

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, state_ = self.memory.sample_buffer(self.batch_size)

            q_next = self.q_target.predict(state_)
            q_eval = self.q_eval.predict(state_)
            q_pred = self.q_eval.predict(state)


    def update(self):
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
        self.update()

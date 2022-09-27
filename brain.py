import numpy as np
import pandas as pd
from math import erf


class RL(object):
    def __init__(self, action_space, learning_rate=0.3, reward_decay=0.9,
                 e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        try:
            self.q_table = pd.read_pickle('q_table.pkl')
            print("Success")
        except:
            print("Fail")
            self.q_table = pd.DataFrame(columns=self.actions)

    def check_state_exist(self, state):
        state = str(state)
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, state):
        state = str(state)
        self.check_state_exist(state)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[state, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))
            action = state_action.astype('float').idxmax()
            pred = state_action.astype('float').max()
        else:
            print("Random")
            action = self.actions[np.random.choice(range(len(self.actions)))]
            pred = 0
        return action, pred

    def learn(self, *args):
        pass


class QTable(RL):
    def __init__(self, actions, learning_rate=0.3, reward_decay=0.8,
                 e_greedy=0.9):
        super(QTable, self).__init__(actions, learning_rate, reward_decay,
                                     e_greedy)

    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table.at[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        self.q_table.at[s, a] += self.lr * (q_target - q_predict)
        self.q_table.to_pickle('q_table.pkl')


class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.3, reward_decay=0.8,
                 e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay,
                                         e_greedy)

    def learn(self, s, a, r, s_, a_):
        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table.at[s, a]
        q_target = r + self.gamma * self.q_table.at[s_, a_]
        self.q_table.at[s, a] += self.lr * (q_target - q_predict)
        self.q_table.to_pickle('q_table.pkl')


class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.4, reward_decay=0.7,
                 e_greedy=0.9, trace_decay=0.8):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate,
                                               reward_decay, e_greedy)
        self.sarsa_lambda = trace_decay
        self.trace = self.q_table.copy()
        self.trace *= 0

    def check_state_exist(self, state):
        state = str(state)
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                [-0.001] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(to_be_append)
            self.trace = self.trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s)
        self.check_state_exist(s_)
        q_predict = self.q_table.at[s, a]
        q_target = r + self.gamma * self.q_table.at[s_, a_]
        error = q_target - q_predict
        self.trace.at[s, a] = 1

        self.q_table += self.lr * error * self.trace
        self.trace *= self.gamma * self.sarsa_lambda
        self.q_table.to_pickle('q_table.pkl')


class LZTable(RL):
    def __init__(self, actions, learning_rate=0.4, reward_decay=0.8,
                 e_greedy=0.8):
        super(LZTable, self).__init__(actions, learning_rate, reward_decay,
                                      e_greedy)
        self.g = [(1 - erf(1.25)) / 2, (erf(1.25) - erf(0.75)) / 2,
                  (erf(0.75) - erf(0.25)) / 2, 3* erf(0.25),
                  (erf(0.75) - erf(0.25)) / 2, (erf(1.25) - erf(0.75)) / 2,
                  (1 - erf(1.25)) / 2]

    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table.at[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        self.q_table.at[s, a] += self.lr * (q_target - q_predict)
        self.q_table.to_pickle('q_table.pkl')

    def choose_action(self, state, score, max_score):
        s = str(state)
        self.check_state_exist(s)

        if np.random.uniform() > 0.95:# or (state[0] > max_score//10 -2 and np.random.uniform() > 0.95) or (state[0] > max_score//10 - 1 and np.random.uniform() > 0.95):
            print("Random")
            return self.actions[np.random.choice(range(len(self.actions)))]

        state_action = self.q_table.loc[s, :]*self.g[3]
        for i in range(0, 7):
            s = str((state[0], state[1] - 3 + i, state[2]))
            if s in self.q_table.index:
                state_action += self.q_table.loc[s, :] * self.g[i]

        """state_action = state_action.reindex(
            np.random.permutation(state_action.index))"""

        return state_action.astype('float').idxmax()


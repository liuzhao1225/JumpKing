import os

# for keras the CUDA commands must come before importing the keras libraries
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import cv2
import time
import pyautogui
from DQN import DDQNAgent
from utils import plotLearning
from screen import WindowCapture
from pynput.keyboard import Key, Controller, Listener

wincap = WindowCapture('跳跃之王')
keyboard = Controller()
QUIT = False
PAUSE = False

def on_press(key):
    try:
        if key.char == 'q':
            global QUIT
            QUIT = True
            print('\nQuit')
        elif key.char == 'p':
            global PAUSE
            if PAUSE:
                PAUSE = False
                print('\nResume')
            else:
                PAUSE = True
                print('\nPause')
    except AttributeError:
        pass


# Collect events until released
listener = Listener(
    on_press=on_press)
listener.start()

### Actions ###
WALK = 0
JUMP = 1
L = 0
R = 1
action_space = []
for i in [0, 1]:
    for j in range(21):
        action_space.append((i, j))

### Images ###
map = cv2.imread('full_map.jpeg')
template = cv2.imread('crouch.png')


### Methods ###

def jump(t):
    keyboard.press(Key.space)
    time.sleep(t / 60)
    keyboard.release(Key.space)
    time.sleep(1.5)

def get_position():
    keyboard.press(Key.space)
    time.sleep(1/30)
    screen = wincap.get_screenshot()
    time.sleep(1/30)
    keyboard.release(Key.space)
    time.sleep(0.2)
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(
        cv2.matchTemplate(map, screen, cv2.TM_CCOEFF_NORMED))

    level = 42 - max_l[1] // 360

    # Find the king

    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    """bbox = max_loc + (24, 24)
    cv2.rectangle(screen, bbox, (255, 0, 255))
    cv2.imshow('window', screen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()"""
    return np.array([level, max_loc[0]+2, max_loc[1]-9])


def get_score(s):
    return 10 + s[0] * 10 - s[2] / 36


#                   tent           sky
trap = np.array([[0, 24, 22], [200, 230, 214]])
def get_reward(s1, s2):
    if any(np.array_equal(t, s2) for t in trap):
        if np.array_equal(s1, s2) or (s1[0] == s2[0] and s1[2] > s2[2]):
            return -5
    if any(np.array_equal(t, s1) for t in trap):
        if s2[0] == s1[0] and s1[2] < s2[2]:
            return 0
    """
    if s1[0] == 3 and s2[0] == 3:
        if s1[2] == 189 and s2[2] == 189:
            return s2[1] - s1[1], score
        if s1[2] == 189 and s2[2] == 45:
            return 10, score
        if s1[2] == 45 and s2[2] == 45:
            return s1[1] - s2[1], score"""
    if np.array_equal(s1, s2):
        return -0.1
    return (s2[0] - s1[0]) * 20 + (s1[2] - s2[2]) / 36


def act(a):
    t = (a[1] +1)/ 30
    d = Key.left
    if a[0] == R:
        d = Key.right


    keyboard.press(Key.space)
    keyboard.press(d)
    time.sleep(t)
    keyboard.release(Key.space)
    keyboard.release(d)
    time.sleep(1+a[1]/25)


def reset():
    observation = wincap.get_screenshot()/255
    observation = cv2.resize(observation, (240, 180))
    return observation, get_position()


def is_complete(s):
    return s[0] == 42 and s[2] < 150


def step(s1, a):
    act(a)
    observation = wincap.get_screenshot()/255
    observation = cv2.resize(observation, (240, 180) )
    s2 = get_position()
    r = get_reward(s1, s2)
    p = get_score(s2)
    complete = is_complete(s2)
    return observation, s2, r, p, complete


def print_action(a):
    direction = 'R'
    t = a[1]
    if a[0] == 0:
        direction = 'L'
    print(direction, ' ', t * 2, ' Frames')

def get_level():
    screen = wincap.get_screenshot()
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(
            cv2.matchTemplate(map, screen, cv2.TM_CCOEFF_NORMED))

    return 42 - max_l[1] // 360

if __name__ == '__main__':

    ddqn_agent = DDQNAgent(alpha=0.001, gamma=0.5,
                           n_actions=len(action_space), epsilon=0.5,
                           epsilon_dec=0.99, epsilon_end=0.01, mem_size=1000,
                           batch_size=32,  input_dims=180*240*1, fname='JK',
                           replace_target=50)
    # n_games = 100
    try:
        ddqn_agent.load_model()
        print("\nSuccessfully Loaded Model")
    except:
        print("\nCreating New Model")

    ddqn_scores = []
    eps_history = []
    # env = wrappers.Monitor(env, "tmp/lunar-lander-ddqn-2",
    #                         video_callable=lambda episode_id: True, force=True)
    done = False
    num_step = 1
    max_score = 0

    for i in range(3):
        print(3 - i)
        time.sleep(1)
    print("\nstart")
    score = 1
    max_score = 1
    observation, state = reset()
    while not QUIT and not done:
        print("\nstep", num_step)
        bbox = (state[1]//2, state[2]//2) + (13, 13)
        s = observation.copy()
        cv2.rectangle(s, bbox, (255, 255, 255))
        cv2.imshow('window', s)
        cv2.waitKey(1)
        action_index = ddqn_agent.choose_action(observation, score, max_score)
        action = action_space[action_index]
        print_action(action)
        observation_, state_, reward, score, done = step(state, action)
        if state_[0] < state[0]:
            level = get_level()
            time.sleep(1)
            level_ = get_level()
            while level < level_:
                time.sleep(1)
                level = level_
                level_ = get_level()
            observation_, state_ = reset()
        reward = get_reward(state, state_)
        score = get_score(state_)
        if score > max_score:
            max_score = score
        print(state, '->', state_, ', reward: %.2f' % reward,
              ', max: %.2f' % max_score)
        ddqn_agent.remember(observation, action_index, reward, observation_,
                            int(done))
        observation = observation_
        state = state_
        ddqn_agent.learn()

        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(max_score)
        if num_step % 100 == 0:
            print('\nSave Model')
            ddqn_agent.save_model()
            filename = 'Jump_King_Learning_curve'
            x = [i + 1 for i in range(len(ddqn_scores))]
            plotLearning(x, ddqn_scores, eps_history, filename)
            print('\nSaved')
        num_step += 1
        while PAUSE:
            time.sleep(1)
    cv2.destroyAllWindows()
    ddqn_agent.save_model()
    print('You can Stop the Program Now')

import numpy as np
from brain import LZTable
from PIL import ImageGrab
from screen import WindowCapture
import cv2
import time
import codecs
# from screen import WindowCapture
# wincap = WindowCapture()
import pyautogui
# from directkeys import PressKey, ReleaseKey, UP, DOWN, LEFT, RIGHT, SPACE, ENTER, ESC
from pynput.keyboard import Key, Controller, Listener

keyboard = Controller()
wincap = WindowCapture('跳跃之王')

QUIT = False
PAUSE = False
speed = 1


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


listener = Listener(on_press=on_press)
listener.start()
### Global Variables ###
WIDTH = 480
HEIGHT = 360
OFF_W = 1040
OFF_H = 530
# OFF_W = 720
# OFF_H = 350
END_W = OFF_W + WIDTH
END_H = OFF_H + HEIGHT
JUMPING = False
q = 0

### States ###

### Actions ###
WALK = 0
JUMP = 1
L = 0
R = 1

### Q-learning ###
LEARNING_RATE = 0.3
DISCOUNT = 0.8
EPISODES = 20000
action_space = []
for i in [0, 1]:
    for j in range(20):
        action_space.append((i, j))

### Images ####
crouch = cv2.imread('crouch.png')
faceLeft = cv2.imread('left.png')
faceRight = cv2.imread('right.png')
map = cv2.imread('full_map.jpeg')
templates = (faceLeft, faceRight)


### Methods ###
def jump(t):
    t = t / speed
    keyboard.press(Key.space)
    time.sleep(t / 60)
    keyboard.release(Key.space)
    time.sleep(1.5 / speed)


def get_state():
    # find the new level
    screen = wincap.get_screenshot()  # cv2.cvtColor(np.array(ImageGrab.grab(bbox = (OFF_W, OFF_H, END_W, END_H))), cv2.COLOR_BGR2RGB)
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(
        cv2.matchTemplate(map, screen, cv2.TM_CCOEFF_NORMED))
    level = 42 - max_l[1] // 360

    # find where the king is
    m = 0
    for template in templates:
        res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > m:
            top_left = max_loc
    bot_right = (top_left[0] + 24, top_left[1] + 24)

    # show image
    """cv2.rectangle(screen, top_left, bot_right, 255, 2)
    cv2.imshow('ha', screen)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()"""

    # quantize the coordinate
    # top_left[0] = top_left[0]//10;
    # top_left[1] = top_left[1]//36;

    #      (          new state             ), reward,    done
    return (level, top_left[0],
            top_left[1])  # , reward, level == 42 and top_left[1] < 180


def complete(state):
    return state[0] == 42 and state[2] < 100


# tent = (0, 96, 122)
# sky = (3, 213, 45)
trap = [(0, 96, 122), (3, 212, 36), (0, 148, 177), (3, 213, 45),
        (3, 209, 22), (5, 352, 5), (8, 352, 5), (8, 199, 182), (0, 136, 302),
        (4, 324, 286), (4, 326, 286)]


def calculate_reward(s0, s1, s2):
    score = s2[0] * 10 + s2[2] / 36
    """if s0 == s1 and s0 == s2:
        return -1000, score
    if s0 == s2:
        return -10, score"""
    if s0 == s2 and s1 == s2:
        return -10, score
    if s2[0] == -1:
        return -1, score

    if s2 in trap:
        if s1 == s2 or (s1[0] == s2[0] and s1[2] > s2[2]):
            return -5, score
    if s1 in trap:
        if s2[0] == s1[0] and s1[2] < s2[2]:
            return 0, score
    """ """
    if s2[0] == 4 and s2[1] < 30 and s2[2] == 214:
        return -5, score
    if s1[0] == 4 and s1[1] < 30 and s1[2] == 214:
        if s2[0] == s1[0] and s1[2] < s2[2]:
            return 0, score
    if s2[0] == 5 and s2[1] > 240 and s2[2] == 158:
        return -5, score
    if s2[0] == 5 and s2[1] > 240 and s2[2] == 118:
        return -10, score
    if s1[0] == 5 and s1[1] > 240 and (s1[2] == 158 or s2[2] == 118):
        if s2[0] == 5:
            return 0, score
    if s2[0] == 6 and s2[2] == 278:
        return -10, score

    '''if s1 == s2:
        return -10, score'''
    reward = (s2[0] - s1[0]) * 20 + (s1[2] - s2[2]) / 36
    if s1[0] == 3 and s2[0] == 3 and s1[2] == 190 and s2[2] == 190:
        if s2[1] < s1[1]:
            reward -= 10
        reward += (s2[1] - s1[1]) / 36
    if s1[0] == 2 and s2[0] == 2 and s1[2] == 278 and s2[2] == 278:
        reward += (s2[1] - s1[1]) / 36
    if s1[0] == 3 and s2[0] == 3 and s1[2] == 46 and s2[2] == 46:
        if s2[1] > s1[1]:
            reward -= 10
        reward += (-s2[1] + s1[1]) / 36
    if s1[0] == 4 and s2[0] == 4 and s1[2] < 70 and s2[2] < 70:
        reward += (-s2[1] + s1[1]) / 36
    if s1[0] == 5 and s2[0] == 5 and s1[2] < 100 and s2[2] < 100:
        reward += (s2[1] - s1[1]) / 36
    if s1[0] == 6 and s2[0] == 6 and s1[2] == 182 and s2[2] == 182:
        reward += (-s2[1] + s1[1]) / 36
        if reward < 0:
            reward *= 10
    if s1[0] == 8 and s2[0] == 8 and s1[1] in range(70, 200) and s2[1] in range(
            70, 200):
        reward = -reward
    if s1[0] == 8 and s2[0] == 8 and s1[2] == 182 and s2[2] == 182:
        reward += (s2[1] - s1[1]) / 48 - 1
    if s1[0] == 14 and s2[0] == 14 and s1[2] == 142 and s2[2] == 142:
        reward += (-s2[1] + s1[1]) / 48
    if s1[0] == 15 and s2[0] == 15 and (s1[2] == 246 and s2[2] == 246) or (
            s1[2] == 286 and s2[2] == 286):
        reward += (s2[1] - s1[1]) / 48 - 1
    if s1[0] == 16 and s2[0] == 16 and s1[2] == 142 and s2[1] > 200 and s2[
        2] > 198:
        reward += 10
    if s1[0] == 16 and s2[0] == 16 and s1[2] == 142 and (
            s2[2] == 142 or s2[2] == 198):
        reward -= 100
    if s1[0] == 17 and s2[0] == 17 and s1[2] == 110 and s2[2] > 110 and s2[1] > \
            s1[1]:
        reward = -reward
    if reward < 0:
        reward *= 2
    reward -= 0.001
    return reward, score


def step(action):
    # take action
    t = action[1] / 30 / speed
    d = Key.left
    if action[0] == R:
        d = Key.right
    wincap.set_front()
    if t == 0:
        keyboard.press(Key.space)
        time.sleep(1 / 30 / speed)
        keyboard.release(Key.space)
        time.sleep(0.3 / speed)
    else:
        if action[1] % 3 == 0:
            keyboard.press(d)
            time.sleep(t)
            keyboard.release(d)
            time.sleep(1 / speed)
        else:
            keyboard.press(Key.space)
            keyboard.press(d)
            time.sleep(t)
            keyboard.release(Key.space)
            keyboard.release(d)
            time.sleep(1.5 / speed)

    # sleep to ensure the animation ends


### Main Program ###

# count down


for i in range(3):
    print(3 - i)
    time.sleep(1)

print("start")
count = 1
RL = LZTable(actions=action_space, learning_rate=0.5, reward_decay=0.9,
             e_greedy=0.9)

max_score = 160
f = codecs.open("max_level.txt", 'w', encoding='utf-8')
f.write('最高：第' + str(int(max_score // 10)) + '层')
f.close()
score = 1
old_state = (0, 0, 0)
state = get_state()
"""while not QUIT:
    step((1, 18))
    state = get_state()
    print(state)"""

action = RL.choose_action(state, 0, max_score)
while not complete(state) and not QUIT:
    #print("step " + str(count))

    """if state[0] == 6 and (state[2] == 182 or state[2] == 126) and state[1] < 275:
        action = (0, 17)"""
    """if state[0] == 2 and state[2] == 278 and state[1] > 242 and state[1] <255:
        action = (1, 2)"""

    # hardcode lv 16
    if state[0] == 16:
        if state[2] == 142:
            if state[1] >= 449:
                action = (0, 17)
            elif state[1] > 438:
                action = (0, 16)
            elif state[1] > 200:
                action = (0, 15)
            else:
                action = (1, 5)
        if state[2] == 230:
            if state[1] >= 241:
                action = (0, 1)
            elif state[1] >= 220:
                action = (0, 10)
            elif state[1] > 150:
                action = (0, 8)
            else:
                action = (0, 18)
        if state[2] == 158:
            action = (0, 18)
        if state[2] == 22:
            action = (1, 5)
        if state[2] == 38:
            action = (0, 18)
    if state == (13, 352, 5):
        action = (1, 1)

    step(action)

    # step(action)
    if state[0] in (14, 16, 17):
        time.sleep(1 / speed)
    new_state = get_state()

    if new_state[0] < state[0]:
        time.sleep(1 / speed)
        new_new_state = get_state()
        while new_new_state[0] < new_state[0]:
            time.sleep(1 / speed)
            new_state = new_new_state
            new_new_state = get_state()

    reward, score = calculate_reward(old_state, state, new_state)

    if score > max_score:
        max_score = score
        f = codecs.open("max_level.txt", 'w', encoding='utf-8')
        f.write('最高：第' + str(int(max_score // 10)) + '层')
        f.close()
    new_action = RL.choose_action(new_state, score, max_score)
    RL.learn(state, action, reward, new_state)

    print((state, action, round(reward, 3), round(max_score, 2)))
    RL.learn(state, action, reward, new_state)
    old_state = state
    state = new_state
    action = new_action
    count += 1
    while PAUSE and not QUIT:
        time.sleep(1)

print("finished")

# Old code
"""
while True:
    screen= cv2.cvtColor(np.array(ImageGrab.grab(bbox = (OFF_W, OFF_H, END_W, END_H))), cv2.COLOR_BGR2GRAY)
    #screen = cv2.GaussianBlur(screen, (5, 5), 0)
    dis = np.sqrt(np.sum(np.square(screen)));
    #img = processImg(screen)
    if dis > 3000:
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(cv2.matchTemplate(map, screen, cv2.TM_CCOEFF_NORMED))
        level = 42 - max_l[1]//360
    max = 0
    top_left = 0
    ind = 0
    i = 0

    for template in templates:
        res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > max:
            ind = i
            top_left = max_loc
        i += 1
    bot_right = (top_left[0] + 24, top_left[1]+24)
    cv2.rectangle(screen, top_left, bot_right, 255, 2)
    cv2.imshow('window', screen)
    print(str(level) + ", " + str(round(dis)));
    lastTime = time.time()

    prevScreen = screen
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
"""

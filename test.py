import cv2
from screen import WindowCapture
import numpy as np
import pyautogui
import os
import time
from pynput.keyboard import Key, Controller, Listener

PATH = 'C:/Users/ZhaoLiu/Desktop/JKData'
wincap = WindowCapture('跳跃之王')
keyboard = Controller()

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


def step(action):
    action -= 0.5
    t = abs(action) * 2
    if action > 0:
        jump_right(t)
    elif action < 0:
        jump_left(t)
    else:
        jump()


def jump_left(t):
    keyboard.press(Key.space)
    keyboard.press(Key.left)
    time.sleep(t)
    keyboard.relesae(Key.space)
    keyboard.release(Key.left)
    time.sleep(1.5)


def jump_right(t):
    keyboard.press(Key.space)
    keyboard.press(Key.right)
    time.sleep(t)
    keyboard.release(Key.space)
    keyboard.release(Key.right)
    time.sleep(1.5)


def jump():
    keyboard.press(Key.space)
    time.sleep(0.1)
    keyboard.release(Key.space)
    time.sleep(0.3)


for i in range(3):
    print(3-i)
    time.sleep(1)


print('start')
max_score = 0
score = 0

while not QUIT:
    im = wincap.get_screenshot()
    cv2.imshow('window', im)
    cv2.imwrite(os.path.join(PATH, 'test.jpg'), im)
    cv2.waitKey(1)
    while PAUSE:
        time.sleep(1)
cv2.destroyAllWindows()

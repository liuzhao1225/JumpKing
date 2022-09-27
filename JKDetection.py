from screen import WindowCapture
import cv2
import time
from pynput.keyboard import Key, Controller

keyboard = Controller()

wincap = WindowCapture('跳跃之王')

map = cv2.imread('full_map.jpeg')
template = cv2.imread('crouch.png')

def get_level(img):
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(
        cv2.matchTemplate(map, img, cv2.TM_CCOEFF_NORMED))

    return 42 - max_l[1] // 360

def get_position():
    keyboard.press(Key.space)
    time.sleep(1/30)
    s = wincap.get_screenshot()
    time.sleep(1/30)
    keyboard.release(Key.space)
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(
        cv2.matchTemplate(s, template, cv2.TM_CCOEFF_NORMED))
    return max_l

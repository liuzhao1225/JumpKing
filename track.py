
import cv2
import numpy as np
from time import time
import pyautogui
from screen import WindowCapture

templates = (cv2.imread('left.png'), cv2.imread('right.png'))
# initialize the WindowCapture class
wincap = WindowCapture('跳跃之王')
tracker = cv2.TrackerCSRT_create()

img = wincap.get_screenshot()
min_v, max_v, min_l, max_l = cv2.minMaxLoc(
    cv2.matchTemplate(img, templates[0], cv2.TM_CCOEFF_NORMED))
bot_right = (max_l[0] + 24, max_l[1] - 24)
bbox = (max_l[0], max_l[1], 24, 24)
tracker.init(img, bbox)
loop_time = time()
while (True):

    # get an updated image of the game
    img_ = wincap.get_screenshot()
    #print(bbox)
    avg = 0
    img = img_.copy()
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(
            cv2.matchTemplate(img, templates[0], cv2.TM_SQDIFF_NORMED))
    avg -= cv2.matchTemplate(img, templates[0], cv2.TM_SQDIFF)
    avg -= cv2.matchTemplate(img, templates[1], cv2.TM_SQDIFF)
    bbox = (min_l[0], min_l[1], 24, 24)
    cv2.rectangle(img, bbox, (255, 0, 0), 2)

    cv2.imshow('sqdiff_normed', img)

    img = img_.copy()
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(
        cv2.matchTemplate(img, templates[0], cv2.TM_CCORR_NORMED))
    avg += cv2.matchTemplate(img, templates[0], cv2.TM_CCORR)
    avg += cv2.matchTemplate(img, templates[1], cv2.TM_CCORR)
    bbox = (max_l) + (24, 24)
    cv2.rectangle(img, bbox, (0, 255, 0), 2)
    cv2.imshow('corr_normed', img)

    img = img_.copy()
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(
        cv2.matchTemplate(img, templates[0], cv2.TM_CCOEFF_NORMED))
    avg += cv2.matchTemplate(img, templates[0], cv2.TM_CCOEFF)
    avg += cv2.matchTemplate(img, templates[1], cv2.TM_CCOEFF)
    bbox = (max_l) + (24, 24)
    cv2.rectangle(img, bbox, (0, 0, 255), 2)
    cv2.imshow('ccoeff_normed', img)
    # debug the loop rate

    # print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()
    img = img_.copy()
    avg /= 6
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(avg)
    bbox = (max_l) + (24, 24)
    cv2.rectangle(img, bbox, (255, 255, 255), 2)
    cv2.imshow('avg', img)
    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

print('Done.')

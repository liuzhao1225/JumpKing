from PIL import ImageGrab, Image, ImageChops
import codecs
import time
from pynput.keyboard import Key, Controller, Listener


keyboard = Controller()
QUIT = False
PAUSE = False
key_down_time = 0.016667
huoyanniao = Image.open('huoyanniao.jpg')
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


def left():
    #print('left')
    keyboard.press('a')
    time.sleep(key_down_time)
    keyboard.release('a')


def right():
    #print('right')
    keyboard.press('d')
    time.sleep(key_down_time)
    keyboard.release('d')


def up():
    #print('up')
    keyboard.press('w')
    time.sleep(5*key_down_time)
    keyboard.release('w')


def down():
    #print('down')
    keyboard.press('s')
    time.sleep(key_down_time)
    keyboard.release('s')


def a():
    #print('A')
    keyboard.press('z')
    time.sleep(key_down_time)
    keyboard.release('z')


def b():
    #print('B')
    keyboard.press('x')
    time.sleep(key_down_time)
    keyboard.release('x')
def f2():
    keyboard.press('f2')
    time.sleep(key_down_time)
    keyboard.release('f2')

def catched():
    im = ImageGrab.grab(bbox=(1500, 350, 1800, 700))
    im.save('this.jpg')
    im = Image.open('this.jpg')
    diff = ImageChops.difference(im, huoyanniao)
    if diff.getbbox():
        print('catched')
        return True
    else:
        print('not catched')
        return False
for i in range(3):
    print(3 - i)
    time.sleep(1)

print('start')
while not catched() and not QUIT:
    f2()
    time.sleep(1)
    if PAUSE:
        time.sleep(1)
print('finished')

# from screen import WindowCapture
# wincap = WindowCapture('VisualBoyAdvance emulator')
from PIL import ImageGrab, Image, ImageChops
import codecs
import time
from pynput.keyboard import Key, Controller, Listener

keyboard = Controller()
QUIT = False
PAUSE = False

key_down_time = 0.016667

normal = Image.open('normal.jpg')

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
    # print('left')
    keyboard.press('a')
    time.sleep(key_down_time)
    keyboard.release('a')


def right():
    # print('right')
    keyboard.press('d')
    time.sleep(key_down_time)
    keyboard.release('d')


def up():
    # print('up')
    keyboard.press('w')
    time.sleep(5 * key_down_time)
    keyboard.release('w')


def down():
    # print('down')
    keyboard.press('s')
    time.sleep(key_down_time)
    keyboard.release('s')


def a():
    # print('A')
    keyboard.press('z')
    time.sleep(key_down_time)
    keyboard.release('z')


def b():
    # print('B')
    keyboard.press('x')
    time.sleep(key_down_time)
    keyboard.release('x')


def get_normal():
    im = ImageGrab.grab(bbox=(1500, 350, 1800, 700))
    im.save('normal.jpg')


def shine():
    im = ImageGrab.grab(bbox=(1500, 350, 1800, 700))
    im.save('this.jpg')
    im = Image.open('this.jpg')
    diff = ImageChops.difference(im, normal)
    if diff.getbbox():
        print('shine')
        return True
    else:
        print('normal')
        return False


def action():
    down()
    time.sleep(0.15)
    up()
    time.sleep(0.1)

    a()
    time.sleep(0.1)
    a()
    time.sleep(0.1)
    a()
    time.sleep(1.3)
    if shine():
        global QUIT
        QUIT = True
        return
    a()
    time.sleep(0.7)
    right()
    time.sleep(0.1)
    down()
    time.sleep(0.1)
    a()
    time.sleep(0.1)
    a()
    time.sleep(0.7)


for i in range(3):
    print(3 - i)
    time.sleep(1)
print('start')
f = open('number.txt', 'r')
n = int(f.read())
f.close()
while not QUIT:
    action()
    f = codecs.open("n.txt", 'w', encoding='utf-8')
    f.write('第' + str(n) + '只')
    f.close()
    n += 1
    f = open('number.txt', 'w')
    f.write(str(n))
    f.close()
    while PAUSE:
        time.sleep(0.1)
print('finished')

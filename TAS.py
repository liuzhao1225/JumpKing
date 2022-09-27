import numpy as np
import pyvjoy
import typing
import os.path
import os
import wmi
import subprocess
from screen import WindowCapture
import cv2
import codecs
import time
from pynput.keyboard import Key, Controller, Listener
keyboard = Controller()
QUIT = False
PAUSE = False
joy = pyvjoy.VJoyDevice(1)
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
wincap = WindowCapture('跳跃之王')
PATH = 'C:\Program Files (x86)\Steam\steamapps\common\Jump King\JumpKing.tas'

DICT = {
    0: ',J\n',
    1: ',L\n',
    2: ',R\n',
    3: ',L,J\n',
    4: ',R,J\n'
}
SPEED = 4

faceLeft = cv2.imread('left.png')
faceRight = cv2.imread('right.png')
JKmap = cv2.imread('full_map.jpeg')
templates = (faceLeft, faceRight)
class Individual:
    def __init__(self, min_len=400, max_len=600):
        self.length = np.random.randint(min_len, max_len)
        self.action_time = np.random.randint(1, 40, self.length)
        self.action = np.random.randint(0, 5, self.length)
        self.wait_time = np.random.randint(1, 50, self.length)

    def mutate(self):
        pos = np.random.randint(0, self.length)
        self.action_time[pos] = np.random.randint(1, 40)
        self.action[pos] = np.random.randint(0, 5)
        self.wait_time[pos] = np.random.randint(1, 50)

    def fitness(self):
        count = 0
        old_top_left = 0
        old_level = 0
        max_score = 0
        #wincap.set_front()
        time.sleep(2)
        button = 1
        joy.set_button(button, 1)
        time.sleep(1)
        joy.set_button(button, 0)
        time.sleep(1)
        start = time.time()
        global QUIT
        while count < 5 and not QUIT:
            screen = wincap.get_screenshot()
            now = time.time()
            min_v, max_v, min_l, max_l = cv2.minMaxLoc(cv2.matchTemplate(JKmap, screen, cv2.TM_CCOEFF_NORMED))
            level = 42-max_l[1]//360
            m = 0
            for template in templates:
                res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val > m:
                    top_left = max_loc
            if old_top_left == top_left and old_level == level:
                count += 1
            elif not count == 0:
                count = 0
            score = (level*20 + max_loc[1]//36)
            print(count)
            old_top_left = top_left
            old_level = level
            if score > max_score:
                max_score = score
        return max_score

    def write_file(self):
        s = '1, J\n60\n1,X\n@23100,30200,1\n'
        for (a, b, c) in zip(self.action_time, self.action, self.wait_time):
            s += str(a)
            s += DICT[b]
            s += str(c)
            s += '\n'
        s += '***'
        s += str(SPEED)
        file = open(PATH, 'w')
        file.write(s)
        file.close()

class Population:
    def __init__(self, pop_size=100, crossover_rate=0.9, min_len=400, max_len=600):
        self.co_rate = crossover_rate
        self.pop_size = pop_size
        self.population = []
        self.max_fitness_history = []
        self.max_fitness = 0
        self.mean_fitness_history = []
        self.generation = 0
        self.min_len = min_len
        self.max_len = max_len
        for i in range(0, pop_size):
            self.population.append(Individual(min_len=min_len, max_len=max_len))

    def reorder(self, index):
        new_population = [0]*self.pop_size
        for i in range(0, self.pop_size):
            new_population[i] = self.population[index[i]]
        self.population = new_population
    def evolve(self):
        fitness = []
        count = 0
        global QUIT
        for individual in self.population:
            individual_fitness = individual.fitness()
            fitness.append(individual_fitness)
            count += 1
            self.individual_info(individual_fitness, count)
            if individual_fitness > self.max_fitness:
                self.max_fitness = individual_fitness
            if QUIT:
                return
        print(fitness)
        index = np.argsort(fitness)
        print(index)
        fitness = np.array(fitness)[index]
        #self.population = np.array(self.population)[index]
        self.reorder(index)
        self.generation += 1
        self.max_fitness_history.append(fitness[0])
        self.mean_fitness_history.append(np.mean(fitness))

        self.population_info()

        self.population = self.population[:self.pop_size//2]
        for i in range(1, self.pop_size//2):
            if(np.random.random() < self.crossover_rate):
                index = np.random.randint(0, i)
                new_individual = self.population[i].crossover(self.population[index])
                self.population.append(new_individual)
        curr_len = len(self.population)
        for i in range(curr_len, self.pop_size):
            self.population.append(Individual(min_len=self.min_len, max_len=self.max_len))
    def crossover(self, father, mother):
        pos = np.random.randint(1, min(father.length, mother.length))
        new_individual = Individual()
        new_individual.action_time = mother[:pos] + father[pos:]
        new_individual.action = mother[:pos] + father[pos:]
        new_individual.wait_time = mother[:pos] + father[pos:]
        new_individual.length = len(new_individual.action)
        return new_individual
    def individual_info(self, fitness, count):
        s = '第' + str(count) + '/' + str(self.pop_size) + '个\n'
        s += '适应度: ' + str(round(fitness, 2))
        print(s)
        f = codecs.open('individual_info.txt', 'w', encoding='utf-8')
        f.write(s)
        f.close()
    def population_info(self):
        s = '第' + str(self.generation) + '代\n'
        s += '最高: ' + str(round(self.max_fitness_history[-1], 2)) + '\n'
        s += '平均: ' + str(round(self.mean_fitness_history[-1], 2))
        print(s)
        f = codecs.open('population_info.txt', 'w', encoding='utf-8')
        f.write(s)
        f.close()



pop_size = 10
min_len = 100
max_len = 200
cross_rate = 0.8
population = Population(pop_size=pop_size, crossover_rate=cross_rate, min_len=min_len, max_len=max_len)
while not QUIT:
    population.evolve()
    while PAUSE:
        time.sleep(1)
print('finished')

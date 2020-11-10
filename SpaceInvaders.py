import pygame
import random
import numpy as np
import cv2
import os.path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten, ZeroPadding2D, UpSampling2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd 
import re
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
import math

pathname = r"D:\OneDrive - Hochschule Albstadt-Sigmaringen\Studium\Semester 5\DesignCPS"
datadirname = "data"
testdirname = "test"
validdirname = "valid"
modeldirname = "model"
datacsvname = "data.csv"
modeljsonname="model-regr.json"
modelweightname="model-regr.h5"
dim = (50,50) 
actionstonum = {"RIGHT": 0,
           "LEFT": 1,
           "SPACE" : 2,
          }
numtoactions = {0: "RIGHT",
           1: "LEFT",
           2: "SPACE",
          }

def create_q_model():
        # Network defined by the Deepmind paper
        inputs = layers.Input(shape=(dim[0], dim[1], 3,))

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(4, activation="linear")(layer5)

        return keras.Model(inputs=inputs, outputs=action)

def run_game(learning_rate = 1.5e-06, epochs = 5, benchmin = 68.0):
    manual = False
    lr = [learning_rate for i in range(epochs)]

    iterations = len(lr)
    benches = []
    qms = []
    qps = []
    counter = 0

    for i in range(iterations):
        print(f"{i}: learning rate: {lr[i]}")
        print(benchmin)
        game = Game(lr[i], "model-regr.h5")
        k = 150 #40
        game.load_replay_memory()
        for j in range(k):
            game.initialize(i, j)
            game.run(j)
        bench, qm, qp = game.print_benchmark()
        benches.append(bench)
        qms.append(qm)
        qps.append(qp)
        game.save_replay_memory()
        game.save_checkpoint(f"model-regr_{i}_{lr[i]:.9f}_{bench:.2f}.h5")
        if bench < benchmin:
            benchmin = bench
            game.save_checkpoint()
        else:
            counter += 1
        if counter == 3:
            counter = 0
            lr *= 0.5 
            
        overallscore = game.print_overall_score()
        overallscores.append(overallscore)
    return benches, qms, qps

model = create_q_model()
model_json = model.to_json()
with open(os.path.join(pathname, modeldirname,modeljsonname), "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(pathname, modeldirname,modelweightname))




class Game:
    screen = None
    aliens = []
    rockets = []
    lost = False

    def __init__(self, width, height, lr=1e-3, checkpointparname="model-regr.h5"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.imgresh1 = None
        self.imgresh2 = None

        self.reward = 0
        self.MAXREWARD = 1.0
        self.PENALTY = -1.0
        self.MOVEPENALTY = 0.0
        
        self.BATCHSIZE = 19
        self.DISCOUNT = 0.99
        self.ALPHA = 0.3
        
        manual=False
        if manual == True:
            self.EPSILON = 0.999
        else:
            self.EPSILON = 0.3
        
        self.REPLAYSIZE = 40_000
        self.overall_score = 0
        self.overall_numbatches = 0
        self.overall_accumulatedstates = np.array([0.0,0.0,0.0,0.0])
        
        
        self.path = os.path.join(pathname, datadirname)
        self.modelpath =  os.path.join(pathname, modeldirname)
        
        self.filename = "data.csv"
        
        self.model = create_q_model()
        self.model_target = create_q_model()

        self.learningrate = lr
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learningrate, clipnorm=1.0)
        self.loss_function = keras.losses.Huber()

        self.checkpointname = os.path.join(pathname, modeldirname,checkpointparname)
        print(f"loading checkpoint: {self.checkpointname}")
        self.model_target.load_weights(self.checkpointname)
        
        self.overall_scores=[]
        self.checkpoint_counter=0
        
        self.shufflelist = []
        self.debugcounter = 0

        done = False

        hero = Hero(self, width / 2, height - 20)
        generator = Generator(self)
        rocket = None

        def run(self, i_index):
            i = i_index + self.get_maxi() + 1
            j = 0
            while not done:
                img1 = np.frombuffer(pygame.image.tostring(self.screen, "RGB"), dtype=np.uint8)
                self.imgresh1 = np.reshape(img1,(self.width,self.height, 3))
                self.imgresh1 = cv2.resize(self.imgresh1, dim, interpolation = cv2.INTER_NEAREST )

                current_state = np.array(self.imgresh1, dtype=np.float32)/255.0
            
                #if len(self.aliens) == 0:
                #    self.displayText("WIN")

                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_LEFT]:  # sipka doleva
                    hero.x -= 2 if hero.x > 20 else 0  # leva hranice plochy
                elif pressed[pygame.K_RIGHT]:  # sipka doprava
                    hero.x += 2 if hero.x < width - 20 else 0  # prava hranice
                elif pressed[pygame.K_q]:
                    pygame.quit()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not self.lost:
                        self.rockets.append(Rocket(self, hero.x, hero.y))

                pygame.display.flip()
                self.clock.tick(60)
                self.screen.fill((0, 0, 0))

                for alien in self.aliens:
                    alien.draw()
                    alien.checkCollision(self)
                    if (alien.y > height):
                        self.lost = True
                        #self.displayText("YOU DIED")

                for rocket in self.rockets:
                    rocket.draw()

                if not self.lost: hero.draw()

                self.write(i,j)

                j+=1

    def write(self, i, j): 

        cv2.imwrite(os.path.join(self.path,"current_{}_{}.png".format(i,j)), self.imgresh1)
        cv2.imwrite(os.path.join(self.path,"next_{}_{}.png".format(i,j)), self.imgresh2)

    def train(self, i, j, term):
        
        # https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/
        
        currentstate = "current_{}_{}.png".format(i,j)

        nextstate = "next_{}_{}.png".format(i,j)      
        
        batch, files = self.pop_batch(self.BATCHSIZE)
        
        assert(self.imgresh1.shape == (dim[0], dim[1],3))
        assert(self.imgresh2.shape == (dim[0], dim[1],3))
        
        batch.append([self.imgresh1, actionstonum[self.changeto], self.reward, self.imgresh2, term, self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]])
        files.append(("current_{}_{}.png".format(i,j), "next_{}_{}.png".format(i,j)))
        
        self.write(i,j)
         
        self.backprop(batch)
        
        self.numbatches += 1
            
        self.push_batch(batch, files)   
  
        return    
    
    def load_replay_memory(self):

        f = open(os.path.join(os.path.join(self.path,datacsvname)), "r")
        
        df = pd.read_csv(f, index_col = 0) 

        for index, row in df.iterrows():

            currentpicname = row["currentstate"]
            action = actionstonum[row["action"]]
            reward = row["reward"]
            nextpicname = row["nextstate"]
            terminated = row["terminated"]

            assert os.path.isfile(os.path.join(self.path,currentpicname)) == True
            assert (action < 5 and action >= 0)
            assert isinstance(reward,int) or isinstance(reward, float)
            assert os.path.isfile(os.path.join(self.path,nextpicname)) == True
            
            self.shufflelist.append([currentpicname,action,reward,nextpicname, terminated])

        random.shuffle(self.shufflelist)

        #print(f"loading: size of replay memory {len(self.shufflelist)}")
        
        f.close()
        
        return

    def displayText(self, text):
        pygame.font.init()
        font = pygame.font.SysFont('Arial', 50)
        textsurface = font.render(text, False, (44, 0, 62))
        self.screen.blit(textsurface, (110, 160))


class Alien:
    def __init__(self, game, x, y):
        self.x = x
        self.game = game
        self.y = y
        self.size = 40

    def draw(self):
        pygame.draw.rect(self.game.screen,  # renderovací plocha
                         (81, 43, 88),  # barva objektu
                         pygame.Rect(self.x, self.y, self.size, self.size))
        self.y += 0.4

    def checkCollision(self, game):
        for rocket in game.rockets:
            if (rocket.x < self.x + self.size and
                    rocket.x > self.x - self.size and
                    rocket.y < self.y + self.size and
                    rocket.y > self.y - self.size):
                game.rockets.remove(rocket)
                game.aliens.remove(self)


class Hero:
    def __init__(self, game, x, y):
        self.x = x
        self.game = game
        self.y = y

    def draw(self):
        pygame.draw.rect(self.game.screen,
                         (210, 250, 251),
                         pygame.Rect(self.x, self.y, 40, 20))


class Generator:
    def __init__(self, game):
        margin = 30  # mezera od okraju obrazovky
        width = 50  # mezera mezi alieny
        for x in range(margin, game.width - margin, width):
            for y in range(margin, int(game.height / 2), width):
                if(random.randint(0,1)==1):
                    game.aliens.append(Alien(game, x, y))
                
                

        # game.aliens.append(Alien(game, 280, 50))


class Rocket:
    def __init__(self, game, x, y):
        self.x = x
        self.y = y
        self.game = game

    def draw(self):
        pygame.draw.rect(self.game.screen,  # renderovací plocha
                         (254, 52, 110),  # barva objektu
                         pygame.Rect(self.x, self.y, 15, 15))
        self.y -= 2  # poletí po herní ploše nahoru 2px/snímek


if __name__ == '__main__':
    game = Game(500, 500)
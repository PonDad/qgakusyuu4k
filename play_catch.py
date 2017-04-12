from keras import backend as K
K.set_image_dim_ordering("th")
assert K.image_dim_ordering() == "th"
from keras.models import Sequential
from keras.layers import Flatten, Dense
from qlearning4k.games import Catch
from keras.optimizers import *
from keras.utils import np_utils
from qlearning4k import Agent

grid_size = 8
hidden_size = 64
nb_frames = 1

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, grid_size, grid_size)))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(3))
model.compile(sgd(lr=.2), "mse")

catch = Catch(grid_size)
agent = Agent(model=model)
model.load_weights('catch.h5')
agent.play(catch)

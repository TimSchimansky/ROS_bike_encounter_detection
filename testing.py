# Author: David Burns
# License: BSD
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from tensorflow.python.keras.layers import Dense, LSTM, Conv1D
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split

from seglearn.datasets import load_watch
from seglearn.pipe import Pype
from seglearn.transform import Segment

import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier


"""def crnn_model(width=100, n_vars=6, n_classes=7, conv_kernel_size=5,
               conv_filters=3, lstm_units=3):
    input_shape = (width, n_vars)
    
    model = Sequential()
    
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                    padding='valid', activation='relu', input_shape=input_shape))
                    
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                     padding='valid', activation='relu'))
                     
    model.add(LSTM(units=lstm_units, dropout=0.1, recurrent_dropout=0.1))
    
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model
"""

class NeuralNet(nn.Module):
    def __init__(self, width=100, n_vars=6, n_classes=7, conv_kernel_size=5, conv_filters=3, lstm_units=3):
        super(NeuralNet, self).__init__()

        self.conv_1 = nn.Conv1d(in_channels=n_vars, out_channels=conv_filters*n_vars, kernel_size=conv_kernel_size, padding='valid')
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv1d(conv_filters*n_vars, conv_filters*conv_filters*n_vars, kernel_size=conv_kernel_size, padding='valid')
        self.relu_2 = nn.ReLU()
        self.lstm_1 = nn.LSTM(input_size=92, hidden_size=5, num_layers=lstm_units, dropout=0.1)
        self.line_1 = nn.Linear(5, n_classes)
        self.soft_1 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv_1(torch.transpose(x, 1,2))
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x, _ = self.lstm_1(x)
        x = self.line_1(x)
        x = self.soft_1(x)
        return x



# load the data
data = load_watch()
X = data['X']
X = [elX.astype(np.float32) for elX in X]
y = data['y']

"""layers = []
layers.append(nn.Conv1d(100, 200, kernel_size=3, padding='valid'))
layers.append(nn.ReLU())
layers.append(nn.Conv1d(200, 400, kernel_size=3, padding='valid'))
layers.append(nn.ReLU())
layers.append(nn.LSTM(2, 10, num_layers=3, dropout=0.1))
layers.append(nn.Softmax())

net = nn.Sequential(*layers)"""

#nn = NeuralNetClassifier(net, max_epochs=10, lr=0.01, batch_size=12, optimizer=torch.optim.RMSprop)

print(1)

# create a segment learning pipeline
pipe = Pype([('seg', Segment(width=100, step=100, order='C')),
             ('crnn', NeuralNetClassifier(NeuralNet, max_epochs=10, lr=0.01, batch_size=12, optimizer=torch.optim.RMSprop))])

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)

print("N series in train: ", len(X_train))
print("N series in test: ", len(X_test))
print("N segments in train: ", pipe.N_train)
print("N segments in test: ", pipe.N_test)
print("Accuracy score: ", score)

img = mpimg.imread('segments.jpg')
plt.imshow(img)

print(1)
import numpy
import string
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = "/content/Famous Five 01 - Five On A Treasure Island By Enid Blyton ( PDFDrive.com ).txt"
text = open(filename).read()
text = text.lower()
text = "".join(v for v in text if v not in string.punctuation)
words =sorted(set(text.split()))

char_to_int = dict((c, i) for i, c in enumerate(words))
int_to_char = dict((i, c) for i, c in enumerate(words))
print(char_to_int)
print(int_to_char)

full_text=text.split()
n_w = len(full_text)
n_uw = len(words)
print ("Total words: ", n_w)
print("Total Unique Words :",n_uw)


seq_length = 20
dataX = []
dataY = []
for i in range(0, n_w - seq_length, 1):
	seq_in = full_text[i:i + seq_length]
	seq_out = full_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)


X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_w)
y = np_utils.to_categorical(dataY)


model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
#model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=100, batch_size=128, callbacks=callbacks_list)

filename = "content/weights.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

import sys
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print( "Reference Sequence:")
print( "\"", ' '.join([int_to_char[value] for value in pattern]), "\"")
print("Predicted:")
for i in range(10):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_w)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write( result + " ")
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
  
  
  

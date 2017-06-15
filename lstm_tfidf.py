'''Train a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

bow_vectorizer = CountVectorizer(ngram_range=(2, 5), max_features=10000)
tfidf = TfidfTransformer()

print('Loading data...')

splits = np.load('data.npz')['splits']

maes = []
rmses = []

for split_index in range(0, len(splits)):
    print("Training %d out of %d" % (split_index + 1, len(splits)))
    (x_train, y_train), (x_test, y_test) = splits[split_index]

    x_train = bow_vectorizer.fit_transform(x_train)
    x_test = bow_vectorizer.fit_transform(x_test)
    x_train = tfidf.fit_transform(x_train)
    x_test = tfidf.fit_transform(x_test)
    x_train = np.array(x_train.todense())
    x_test = np.array(x_test.todense())
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    maxlen = x_train.shape[0]
    max_features = x_train.shape[2]

    print('x_test shape:', x_test.shape)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(1, max_features)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mse'])

    print('Train...')
    history = model.fit(x_train, y_train,
              batch_size=32,
              epochs=8,
              validation_data=(x_test, y_test))

    maes.append(history.history['val_loss'][-1])
    rmses.append(np.sqrt(history.history['val_mean_squared_error'][-1]))

print(maes)
print("MAE: ", np.array(maes).mean())
print("RMSE: ", np.array(rmses).mean())

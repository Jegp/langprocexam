import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

splits = np.load("data.npz")["splits"]

maes = []
r2s = []
mrses = []

for split_index in range(0, len(splits)):
    print("Training %d out of %d" % (split_index + 1, len(splits)))
    (x_train, y_train), (x_test, y_test) = splits[split_index]

    pipeline = Pipeline([('bow', CountVectorizer(ngram_range=(2, 5), max_features=10000)),
                         ('tfidf', TfidfTransformer()),
                         ('linear', LinearRegression())])

    pipeline.fit(x_train, y_train)
    
    prediction = pipeline.predict(x_test)
    maes.append(metrics.mean_absolute_error(y_test, prediction))
    r2s.append(metrics.r2_score(y_test, prediction))
    mrses.append(np.sqrt(metrics.mean_squared_error(y_test, prediction)))

print(maes)
print("MAE: ", np.array(maes).mean())
print("R2: ", np.array(r2s).mean())
print("RMSE: ", np.array(mrses).mean())

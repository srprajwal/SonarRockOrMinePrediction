
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('/content/sonarData.csv',header=None)

sonar_data.head()

sonar_data.shape

sonar_data.describe()

sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_pre = model.predict(X_train)
training_data_acc = accuracy_score(X_train_pre, Y_train)

print(training_data_acc)

X_test_pre = model.predict(X_test)
test_data_acc = accuracy_score(X_test_pre, Y_test)

print(test_data_acc)

input_data = (0.0209,0.0191,0.0411,0.0321,0.0698,0.1579,0.1438,0.1402,0.3048,0.3914,0.3504,0.3669,0.3943,0.3311,0.3331,0.3002,0.2324,0.1381,0.3450,0.4428,0.4890,0.3677,0.4379,0.4864,0.6207,0.7256,0.6624,0.7689,0.7981,0.8577,0.9273,0.7009,0.4851,0.3409,0.1406,0.1147,0.1433,0.1820,0.3605,0.5529,0.5988,0.5077,0.5512,0.5027,0.7034,0.5904,0.4069,0.2761,0.1584,0.0510,0.0054,0.0078,0.0201,0.0104,0.0039,0.0031,0.0062,0.0087,0.0070,0.0042)

input_data_asnumpy_array = np.asarray(input_data)

input_data_reshape = input_data_asnumpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshape)

print(prediction)


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

print(X)
print(y)

clf = Perceptron(shuffle=True,random_state=42,verbose=1,tol=0.0001)
clf.fit(X,y)

print(clf.coef_)
print(clf.intercept_)

y = np.array([0, 1, 1, 0])

print(X)
print(y)

clf = MLPClassifier(hidden_layer_sizes=(2,),max_iter=5000,random_state=44,verbose=2,tol=0.0000001)
clf.fit(X,y)

print(clf.coefs_)
print(clf.intercepts_)

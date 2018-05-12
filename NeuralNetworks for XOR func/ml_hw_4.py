import numpy as np
from matplotlib import pyplot as plt

X = np.array([[1, 0, 0],
              [1, 1, 0],
              [1, 0, 1],
              [1, 1, 1]]) 
y = np.array([[0], [1], [1], [0]]).T
n_epoch = 10000
lr = 0.2

losses = []

sigmoid = lambda x : 1/ (1 + np.exp(-x))
dsigmoid = lambda x : x * (1-x) 
loss = lambda h, t : 0.5*np.dot((h-t).T, (h-t)) 
predict = lambda x : 1 if sigmoid(x) > 0.5 else 0

#np.random.seed(10)

w1 = 2*np.random.random((2, 3)) - 1
w2 = 2*np.random.random((1, 3)) - 1

for i in range(n_epoch):
    """
    Forward-Pass:
        1 - receive data from previous layer
        2 - add bais to it (bais is constant its value determine in weight matrix)
        3 - multiply with the weight matrix of the current layer
        4 - take the sigmoid of the result
        5 - feed it to the next layer
    """
    x_0 = X.T
    a_1 = np.dot(w1, x_0)
    x_1 = np.vstack((np.ones((1,x_0.shape[1])), sigmoid(a_1))) # adding bais for the hidden layer
    a_2 = np.dot(w2, x_1)
    x_2 = sigmoid(a_2) # we are using sigmoid because without it gives over-flow error
    
    losses.append(np.mean(loss(x_2, y)))
    
    """
    Backward-Pass:
    NOTE : out weight matrix index start from 1, so the layer
        1 - compute the derivative of the cost respect to next layer input, d(loss)/d(a_i)
        2 - compute the derivative of the next layer input respect to previous layers weight matrix, d(a_i)/d(w_i) 
        3 - multiply the result of 1 and 2, d(loss)/d(a_i) * d(a_i)/d(w_i) = d(loss)/d(w_i)
    """
    # NOT : baises included in the weights
    d_2 = (x_2 - y) * dsigmoid(x_2) # d(loss)/d(x_2) * d(x_2)/d(a_2) = d(loss)/d(a_2)
    d_1 = np.dot(w2[:, 1:].T, d_2) * dsigmoid(x_1[1:]) # d(loss)/d(a_2) * d(a_2)/d(x_1) * d(x_1)/d(a_1) = d(loss)/d(a_1)
    
    dw2 = np.dot(d_2, x_1.T) # d(loss)/d(a_2) * d(a_2)/d(w_2) = d(loss)/d(w_2)
    dw1 = np.dot(d_1, x_0.T) # d(loss)/d(a_1) * d(a_1)/d(w_1) = d(loss)/d(w_1)
    
    # we are computing the mean deltas 
    w2 -= lr * dw2 / x_0.shape[1] 
    w1 -= lr * dw1 / x_0.shape[1]
    
    
print("Prediction : \n", x_2.T)
    
plt.plot(losses)
plt.title("Loss")
plt.show()


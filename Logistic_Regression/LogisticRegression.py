import numpy as np

def read(filename):
    X = []
    y = []
    label = 0
    
    for i, digit in enumerate(open(filename, "r").read().splitlines()):
        num = digit.split(" ")
        x = np.array([int(n) for n in num if not n == ""])
        X.append(x/6.0) #Normalizing the data!
        y.append(label)
        
        if (i+1) % 200 == 0:
            label += 1
    
    return np.array(X), np.array(y)

def process_data(X, y):
    X_train = np.zeros((X.shape[0]//2, X.shape[1]))
    X_test = np.zeros((X.shape[0]//2, X.shape[1]))
    y_train = np.zeros(y.shape[0]//2)
    y_test = np.zeros(y.shape[0]//2)

    for i in range(10): #Since this LR we have is we iterat over all the data we doesn't need to shuffle
        X_train[i*100:(i+1)*100, :] = X[i*200:i*200 + 100, :]
        X_test[i*100:(i+1)*100, :] = X[i*200+100:i*200+200, :]
        y_train[i*100:(i+1)*100] = y[i*200:i*200 + 100]
        y_test[i*100:(i+1)*100] = y[i*200+100:i*200+200]

    return X_train, y_train, X_test, y_test

class LogisticRegression:
    def __init__ (self, lr = 10e-1, max_iter = 100, l =10e-2, random_seed = None):
        self.epoch = max_iter
        self.lr = lr
        self.l = l
        self.theta = None
        self.labels = None
        self.random_seed = random_seed
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def gradient(self, X, y, theta):
        h = self.sigmoid(np.dot(X, theta.T)) # A * x = b
        m = y.shape[0]
        
        tmp = np.zeros(theta.shape)
        tmp[1:] = theta[1:]
        
        return (np.dot(X.T, h-y) + self.l*tmp) / m
    
    def cost(self, X, y_):
        
        J = []
        
        for i in range(self.theta.shape[1]):
            y = np.array([1 if label == self.labels[i] else 0 for label in y_])
            theta = self.theta[:,i].T
            h = self.sigmoid(np.dot(X, theta))
            m = y.shape[0]
            
            first = np.dot(-y.T, np.log(h)) # if y = 1
            second = np.dot((1-y).T, np.log(1-h)) # if y = 0
            reg = np.dot(theta.T, theta.T)
            J.append((first - second)/m + reg*self.l/(2*m))
        
        return J
    
    
    def predict(self, X):
        return self.sigmoid(np.dot(X, self.theta))

    
    def percentage(self, X, y):
        correct = 0
        
        for i in range(y.shape[0]):
            p = self.predict(X[i,:]).argmax(0)
            if y[i] == self.labels[p]:
                correct += 1
        
        return correct/y.shape[0]
    
    def fit(self, X, y):
    	if 	self.random_seed:
    		np.random.seed(self.random_seed)
    	self.labels = np.unique(y)
    	n_classes = self.labels.shape[0]
    	self.theta = 4*np.random.rand(X.shape[1], n_classes) - 2
    	for idx, c in enumerate(self.labels):
    		y_ = np.array([1 if i == c else 0 for i in y])
    		theta = self.theta[:,idx]
    		for i in range(self.epoch):
    			g = self.gradient(X, y_, theta)
    			theta -= self.lr*g
    		self.theta[:,idx] = theta

if __name__ == '__main__':
	X, y = read("mfeat-pix.txt")
	X_train, y_train, X_test, y_test = process_data(X, y)
	X_train = np.insert(X_train, 0, values=np.ones(X_train.shape[0]), axis=1)
	X_test = np.insert(X_test, 0, values=np.ones(X_test.shape[0]), axis=1)
	l = LogisticRegression(lr = 1.5, max_iter = 1000, l =1)
	l.fit(X_train, y_train)
	print("Train ACC. ", l.percentage(X_train, y_train), " Test ACC. ", l.percentage(X_test, y_test))
	

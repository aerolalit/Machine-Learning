#Import dependecies
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


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

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.fisher_faces = None
        self.n_classes = None
    
    def process(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        l_i = dict(zip(unique, counts))
        self.n_classes = unique.shape[0]
        
        data = {}
        for label in unique:
            data[label] = []
        
        for x, label in zip(X, y):
            data[label].append(x)
        
        total_m = 0
        for label in data:
            d = np.array(data[label])
            mean = np.array(np.mean(d, axis=0))
            
            
            l_ = l_i[label]
            
            means = np.repeat(mean, l_, axis=0)
            means = means.reshape(mean.shape[0], l_)
           
            d = d.T - means
           
            
            data[label] = {"L": d,"mean": mean, "l": l_}
            total_m += mean*l_
        
        total_m /= y.shape[0]
        
        return data, total_m.T
        
        
    def fit(self, X, y):
        data, mean = self.process(X, y)
        self.mean = mean
        S_B = [] # scatter between the classes (max this)
        S_W = [] # scatter in the classes (min this)
        
        for c in data:
            S_B.append(data[c]["mean"] - mean) 
            S_W.append(np.dot(data[c]["L"], data[c]["L"].T)) # Scatter In the classes
            
        S_B = np.array(S_B).T
        
        
        S_W = sum(S_W)  
        S_B = np.dot(S_B, S_B.T) # Scatter between classes
        
        W = np.dot(LA.pinv(S_W), S_B) # Solving S_B*u = S_w*a*u => pinv(S_W)*S_B*u = a*u, u is the eig. vec. of W
        [eigvals, eigvecs] = LA.eig(W)

        # Note eig. vecs and vals. complex so we only take the real part
        eiglist = [[eigvals[i].real, eigvecs[:, i].real] for i in range(len(eigvals)) if eigvals[i] >= 1e-4] 

        # sort the eigvals in decreasing order
        eiglist = sorted(eiglist, key = lambda x : x[0], reverse = True)

        # take the first num_dims eigvectors
        if len(eiglist) < self.n_components:
            self.fisher_faces = np.array([eiglist[i][1] for i in range(len(eiglist))])
        else:
            self.fisher_faces = np.array([eiglist[i][1] for i in range(self.n_components)])
      
      
    def transform(self, X):

        return np.dot(X, self.fisher_faces.T)  

if __name__ == '__main__':
    import matplotlib.cm as cm



    X, y = read("mfeat-pix.txt")
    lda = LDA(2)
    lda.fit(X,y)
    x_lda = lda.transform(X)

    colors = cm.rainbow(np.linspace(0, 1, 10))
    for i in range(10):
        plt.scatter(x_lda[200*i:200*(i+1), 0], x_lda[200*i:200*(i+1), 1], color=colors[i], label=str(i))
    plt.title("LDA")
    plt.legend()
    plt.show()
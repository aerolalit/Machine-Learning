'''
	Machine Learning 
	Programming Assignment 2
	Group Members:-
	Kushal Dhungana
	Alkim Alkun
	Cem Altun
'''


# Import dependecies
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys

def read(filename):
    X = []
    y = []
    label = 0

    for i, digit in enumerate(open(filename, "r").read().splitlines()):
        num = digit.split(" ")
        x = 6 - np.array([int(n) for n in num if not n == ""])
        X.append(x/6.0) # Normalizing the data
        y.append(label)

        if (i+1) % 200 == 0:
            label += 1
    
    return np.array(X), np.array(y)


class PCA:
	def __init__(self, n_components):
	    self.n_components = n_components
	    self.eigen_faces = None
	    self.covs = None
	    self.mean = None
	    
	    
	def fit(self, X, y=None):
	    mean = np.array([np.mean(X, axis=0)])
	    means = np.repeat(mean, X.shape[0], axis = 0)
	    self.mean = mean
	    
	    X_ = X - means # Centered the Data
	    C = np.dot(X_.T, X_)
	    [U, S, V] = LA.svd(C) # Apperantly C is not the Covariance matrix of X_
	    
	    self.covs = []
	    total = np.sum(S)
	    until = S.shape[0]

	    # Decide until where we take the eigen vectors 
	    if S.shape[0] > X.shape[1]:
	        until = X.shape[1]

	    for i in range(1, until):
	        self.covs.append(np.sum(S[:i])/total)
	       

	    closest = 1000
	    index = 0
	    # Find the closest matching value k, if n_component given as percentage
	    if isinstance(self.n_components, float):
	        for i, cov in enumerate(self.covs):
	            tmp = abs(cov - self.n_components)
	            if tmp < closest:
	                closest = tmp
	                index = i+1
	        self.eigen_faces = V[:,:index]
	        self.n_components = index
	    
	    # Take top n_component eigen vectors
	    if isinstance(self.n_components, int):
	        self.eigen_faces = V[:][:self.n_components]

	    
	def transform(self, X):

	    X_centered = X - np.repeat(self.mean, X.shape[0], axis = 0)
	    return np.dot(X_centered, self.eigen_faces.T) 

	def retransform(self, X):
		
	    return np.dot(X, self.eigen_faces)  +  np.repeat(self.mean, X.shape[0], axis = 0)


if __name__ == '__main__':
	X, y = read("mfeat-pix.txt")
	digit = int(sys.argv[1]) # First, Enter Your digit
	X_ = X[digit*200:digit*200 + 200]

	pca = PCA(float(sys.argv[2])) # Second, Enter Your percentage that you want to shrink
	pca.fit(X_)
	X_pca = pca.transform(X_)
	
	X_constructed = pca.retransform(X_pca)

	print("Number of eigen vectors used : ", pca.n_components)


	# Plot some images
	f, ax = plt.subplots(2,5, figsize=(16, 15))
	for i in range(5):
		ax[0,i].imshow(X_[i,:].reshape(16,15), cmap="gray") 
		ax[1,i].imshow(X_constructed[i,:].reshape(16,15), cmap="gray") 
	f.suptitle("Reconstructed Images", fontsize=12)
	plt.show()

	# Plot top 5 eigen vectors Note if percentage of shrinking a lot it will not use top 5 eigen vectors
	if pca.n_components < 5:
		if pca.n_components != 1:
			f2, ax2 = plt.subplots(1, pca.n_components, figsize=(16, 15))
			for i in range(pca.n_components):
				ax2[i].imshow(pca.eigen_faces[i,:].reshape(16,15), cmap="gray")
		else:
			f2, ax2 = plt.subplots(1, figsize=(16, 15))
			ax2.imshow(pca.eigen_faces.reshape(16,15), cmap="gray")

	else:
		f2, ax2 = plt.subplots(1, 5, figsize=(16, 15))
		for i in range(5):
			ax2[i].imshow(pca.eigen_faces[i,:].reshape(16,15), cmap="gray")	
	f2.suptitle("Top eigen vectors", fontsize=12)
	plt.show()

	plt.figure(3)
	plt.plot(pca.covs)
	plt.title("sum i to m (cov) / sum total (cov)")
	plt.show()
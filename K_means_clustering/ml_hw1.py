'''
	Machine Learning 
	Programming Assignment 1
	Group Members:-
	Kushal Dhungana
	Alkim Alkun
	Cem Altun
'''

#Import dependecies
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys
from random import sample, seed


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

class Kmean:
	def __init__ (self, n_centroids, threshold = 0.001, epoch = 100):
		"""
			n_centriods : number of codebook vectors
			threshold 	: for checking changes between old and new codebook vectors
			epoch 		: maximum number of iterations
		"""

		self.n_centroids = n_centroids
		self.clusters = None 
		self.centroids = None
		self.threshold = threshold
		self.epoch = epoch
	    
	def cost(self, X):
	   	#Calculates the average cost!

		cost = 0
		for c in range(self.n_centroids):
			for i in self.clusters[c]:
				cost += LA.norm(X[i] - self.centroids[c])
		return cost/X.shape[0]

	def InitCentroids(self, X):
		row, col = X.shape

		#Option 1
		#indexes = sample(range(row), self.n_centroids)
		#self.centroids = np.array([X[i] for i in indexes])

		#Option 2
		self.centroids = np.random.rand(self.n_centroids, col)


	def findClosestCentroids(self, X):
		row, col = X.shape
		sum_clusters = np.zeros((self.n_centroids,col))
		idx_clusters = [[] for _ in range(self.n_centroids)]

		for i in range(row):
		    shortest = 1e10
		    index = -1
		    x = X[i]
		    for c in range(self.n_centroids):
		        distance = LA.norm(x-self.centroids[c])
		        if distance < shortest:
		            shortest = distance
		            index = c
		    idx_clusters[index].append(i)
		    sum_clusters[index] += x #We don't need to compute sum of clusters again, 
		    							#because we are already calculating here!

		#Removing centroids with empty cluster!
		for i in range(self.n_centroids-1,-1,-1):
		    if not idx_clusters[i]:
		        del idx_clusters[i]
		        sum_clusters = np.delete(sum_clusters, i, 0)
		        self.centroids = np.delete(self.centroids, i, 0)
		        self.n_centroids -= 1

		self.clusters = idx_clusters
		return sum_clusters
	
	#Compute Means
	def computeMeans(self, X, sum_clusters):
	    prev_centroids = np.copy(self.centroids)
	    
	    for c in range(self.n_centroids):
	        n = len(self.clusters[c])
	        self.centroids[c] = sum_clusters[c]/n
	        
	    return prev_centroids
	    
	def checkChanges(self, prev_centroids):
	    for c in range(self.n_centroids):
	        if LA.norm(prev_centroids[c]-self.centroids[c]) > self.threshold:
	            return True
	    return False
	    
	#--------MAIN FUNCTION------#      
	def fit(self, X, y=None):
	    self.InitCentroids(X)
	    
	    for i in range(self.epoch):
	        print("Iteration : " + str(i), end=" | ")
	        
	        sum_clusters = self.findClosestCentroids(X)
	        prev_centroids = self.computeMeans(X, sum_clusters)

	        print("Avg. Cost : " + str(self.cost(X)))
	        if not self.checkChanges(prev_centroids):
	            break

if __name__ == '__main__':
	#Removing RNG factor!
	#np.random.seed(0)  #Initializing the seed.
	#seed(0) 
	X, y = read("mfeat-pix.txt")
	n_centroids = int(sys.argv[2]) # Second, Enter Your number of centroids
	digit = int(sys.argv[1]) # First, Enter Your digit
	cl = Kmean(n_centroids)
	cl.fit(X[digit*200:digit*200 + 200])

	f, ax = plt.subplots(1,cl.n_centroids, figsize=(20, 10)) 

	for i in range(cl.n_centroids):
		ax[i].imshow(cl.centroids[i].reshape(16,15), cmap="gray")  # for k=1 we delete the [i] from ax[i].
	f.suptitle("centroids", fontsize=12)
	plt.show()



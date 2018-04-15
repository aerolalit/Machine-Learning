# Import dependecies
import numpy as np
import numpy.linalg as LA


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

class LinearRegression:
    def __init__(self):
        self.theta = None
    
    def fit(self, X, y):
        #pseudo inverse
        #self.theta = np.dot(LA.pinv(X), y) # Another alternative!
        self.theta = np.dot(LA.pinv(np.dot(X.T, X)), np.dot(X.T, y)) #When we add bais matrix become singular!
    
    def predict(self, x):
        return np.dot(x, self.theta).argmax(0) # only works for vectors
    
    def avgcost(self, X, y):
        cost = 0
        
        for i, x in enumerate(X):
            p = np.dot(x, self.theta)
            cost += np.dot((y[i,:]-p).T,(y[i,:]-p))
            
        return cost / y.shape[0]
    
    def percentage(self, X, y):
        
        correct = 0
        
        for i, x in enumerate(X):
            if y[i, self.predict(x)] == 1:
                correct += 1
        
        return 100*(correct/y.shape[0])


def process_data(X, y):
    y_ = []
    for i in range(y.shape[0]):
        tmp = np.zeros((10,))
        tmp[y[i]] = 1
        y_.append(tmp)
    y_ = np.array(y_)
    X_train = np.zeros((X.shape[0]//2, X.shape[1]))
    X_test = np.zeros((X.shape[0]//2, X.shape[1]))
    y_train = np.zeros((y_.shape[0]//2, y_.shape[1]))
    y_test = np.zeros((y_.shape[0]//2, y_.shape[1]))

    for i in range(10): #Since this LR we have is not iterative data doesn't need to be sorted
        X_train[i*100:(i+1)*100, :] = X[i*200:i*200 + 100, :]
        X_test[i*100:(i+1)*100, :] = X[i*200+100:i*200+200, :]
        y_train[i*100:(i+1)*100, :] = y_[i*200:i*200 + 100, :]
        y_test[i*100:(i+1)*100, :] = y_[i*200+100:i*200+200, :]

    return X_train, y_train, X_test, y_test

def plot_data(X_test, y_test, X_train, y_train, jump = 10):

    x_train_c = []
    x_train_p = []
    
    x_test_c = []
    x_test_p = []
    
    y_axis = []
    
    lr = LinearRegression()
    
    total = X_all_train.shape[1]
    
    #Adding 2-features
    lr.fit(X_train[:,:2], y_train)
    x_train_c.append(lr.avgcost(X_test[:,:2], y_test))
    x_test_c.append(lr.avgcost(X_train[:,:2], y_train))
    x_train_p.append(lr.percentage(X_train[:,:2], y_train))
    x_test_p.append(lr.percentage(X_test[:,:2], y_test))
    y_axis.append(2)
    for i in range(1, total//jump):
    	lr.fit(X_train[:,:i*jump], y_train)
    	x_test_c.append(lr.avgcost(X_test[:,:i*jump], y_test))
    	x_train_c.append(lr.avgcost(X_train[:,:i*jump], y_train))
    	x_train_p.append(lr.percentage(X_train[:,:i*jump], y_train))
    	x_test_p.append(lr.percentage(X_test[:,:i*jump], y_test))    
    	y_axis.append(i*jump)

    print("Min. Cost : ", min(x_test_c), "Max. Acc. : ", max(x_test_p), "Position : ", x_test_p.index(max(x_test_p)))
    #x_test_c = np.log(np.array(x_test_c))
    #x_train_c = np.log(np.array(x_train_c))


    fig, ax = plt.subplots(1,2, figsize=(20, 10))

    ax[0].plot(y_axis, x_train_c, "r")
    ax[0].plot(y_axis, x_test_c, "b")
    ax[0].set_title("Avg. Cost")
    ax[0].legend(["Train", "Test"])
    ax[0].set_ylabel("Cost")
    ax[0].set_xlabel("Iteration")
    
    ax[1].plot(y_axis, x_train_p, "r")
    ax[1].plot(y_axis, x_test_p, "b")
    ax[1].set_title("Accuracy")
    ax[1].legend(["Train", "Test"])
    ax[1].set_ylabel("Percentage(%)")
    ax[1].set_xlabel("Iteration")

    plt.suptitle("Number of features " + str(total))
    plt.show()

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.split(os.getcwd())[0]) 

    X, y = read("mfeat-pix.txt")
    X_train, y_train, X_test, y_test = process_data(X, y)
    from K_means_clustering.Kmeans import Kmean
    from Principal_Component_Analysis.PCA import PCA
    from Linear_Discriminant_Analysis.LDA import LDA

    y_train_ = np.zeros((y.shape[0]//2,))
    y_test_ = np.zeros((y.shape[0]//2,))
    for i in range(10):
        y_train_[i*100:(i+1)*100] = y[i*200:i*200 + 100]
        y_test_[i*100:(i+1)*100] = y[i*200+100:i*200+200]

    lda = LDA(10)
    lda.fit(X_train, y_train_)
    X_lda_train = lda.transform(X_train)
    X_lda_test = lda.transform(X_test)

    pca = PCA(1.0)
    pca.fit(X_train)
    X_pca_train = pca.transform(X_train)
    X_pca_test = pca.transform(X_test)

    km = Kmean(200, epoch = 15) #Kmeans takes to much time to fit
    km.fit(X_train)
    X_train_kmeans = km.transform(X_train)
    X_test_kmeans = km.transform(X_test)

    X_all_train = np.concatenate((np.zeros((X_train.shape[0],1)), X_lda_train, X_train_kmeans, X_pca_train, X_train), axis=1)
    X_all_test = np.concatenate((np.zeros((X_test.shape[0],1)), X_lda_test, X_test_kmeans, X_pca_test, X_test), axis=1)


    import matplotlib.pyplot as plt
    import matplotlib

    SIZE = 15
    matplotlib.rc('font', size=SIZE)
    matplotlib.rc('axes', titlesize=SIZE)

    plot_data(X_all_test, y_test, X_all_train, y_train)
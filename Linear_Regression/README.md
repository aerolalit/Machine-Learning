Experiment
==========
The optimized model ` W_{opt} ` is constructed from 100 samples of training images, of each digits 0-9, using feature vectors `m = 1:240` . Then, the number of missclassification for training and testing set of images is calculated and plotted for each m = 1:240.

Observation
------------
*	For training set of images, the number of missclassifications (indicated by small red circles in graph below,) decreases with the increase of number of feature vectors taken into account.
*	However, for testing set of images, the number of missclassification (indicated by small blue circles in graph below,) decreases first and then later it increseas  with the increase in the number of feature vectors taken into account. \\
`The number of missclassification rate is minimum for m ≈ 50.`
![alt text](https://github.com/aerolalit/Machine-Learning/blob/master/Misclassifications/Graph.png)

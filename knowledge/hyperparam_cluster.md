### Performing k-Means with 3 clusters using default settings for other parameters.
```R
kmeans_result <- kmeans(data, centers = 3)
```

### k-Means with an increased maximum number of iterations
Increasing the maximum number of iterations to 100. This gives k-Means more time to converge, especially useful for more complex datasets or when the default iterations do not lead to convergence.
```R
kmeans_result_maxiter <- kmeans(data, centers = 3, iter.max = 100)
```
### k-Means with a different initialization method (k-means++)
Running the k-Means algorithm with 25 different starting configurations, 
increasing the probability of finding a global optimum and avoiding local minima.
```R
kmeans_result_plusplus <- kmeans(data, centers = 3, nstart = 25)
```
### Standard GMM with 3 clusters
```R
gmm_result <- Mclust(data, G = 3)
```
### GMM with variable covariance structure
Applying GMM with fully variable covariance structure ('VVV'), giving each cluster its own volume, shape, and orientation parameters.
```R
gmm_result_vcv <- Mclust(data, G = 3, modelNames = "VVV")
```
### GMM with the assumption that all clusters have the same shape
The 'EEE' model assumes all clusters have the same shape but can have different volumes, 
useful when assuming clusters are similar in shape but can vary in size.
```R
gmm_result_e <- Mclust(data, G = 3, modelNames = "EEE")
```
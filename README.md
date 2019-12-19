# Regression & Clustering
Project to learn about Prediction techniques using Regression & Clustering algorithms

## Project 1:
Project to learn on how to create a model for weight prediction based on diet and person information
Steps:
* Load the diet dataset
* Split the dataset into test and train datasets; 70% of the data should be used for training the model and the rest for testing
* Train a LinearRegression regression model by fitting on the train dataset;
* Based on the trained model, predict the weights of people in the test dataset;
* Print the predictions and the real weights
* Print the mean square error for your predictions

## Project 2:
Project to learn on clustering using K-Means split the iris dataset into 3 clusters
Steps:
* Load the diet dataset
* Drop the 'species' column; this is required because clustering is an unsupervised method.
* Use K-means to cluster the data into 3 clusters; because we know that there are 3 different species of flowers in this dataset
* Plot the clusters based on what you have learnt in Visualisation Lab. Plot a scatter chart using x=petal_length', y='petal_width' for each cluster
* Label each data point with the true label of flower class.

## Project 3:
Project to learn on clustering using AgglomerativeClustering split the diet dataset into 3 clusters based on the diet types
Steps:
* Load the diet dataset
* Drop the 'Diet' column; this is required because clustering is an unsupervised method.
* Use AgglomerativeClustering to cluster the data into 3 clusters; because we know that there are 3 different types of diet in this dataset
* Plot the clusters based on what you have learnt in Visualisation Lab. Plot a scatter chart using x=pre.weight', y='weight6weeks' for each cluster
* Label each data point with the true label of diet.
* Change the Clustering algorithm to KMeans; which one is better for this problem?
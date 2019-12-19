import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


if __name__ == '__main__':
    df = pd.read_csv('iris.csv')
    df = shuffle(df)
    # Split the data into test and train parts
    df_x = df.drop('species', axis=1)

    # Fit a k-means estimator
    estimator = KMeans(n_clusters=3)
    estimator.fit(df_x)
    # Clusters are given in the labels_ attribute
    labels = estimator.labels_
    df['cluster'] = pd.Series(labels, index=df.index)

    print(labels)
    # divide the dataset into three dataframes based on the species
    cluster_0_df = df.query('cluster == 0')
    cluster_1_df = df.query('cluster == 1')
    cluster_2_df = df.query('cluster == 2')

    fig, axes = plt.subplots(nrows=1, ncols=1)

    ax = cluster_0_df.plot.scatter(x='petal_length', y='petal_width', label='Cluster-0', color='blue', ax=axes)
    ax = cluster_1_df.plot.scatter(x='petal_length', y='petal_width', label='Cluster-1', color='red', ax=ax)
    ax = cluster_2_df.plot.scatter(x='petal_length', y='petal_width', label='Cluster-2', color='green', ax=ax)

    for i, label in enumerate(df['species']):

        label = label[0:4]
        ax.annotate(label, (list(df['petal_length'])[i], list(df['petal_width'])[i]), color='gray', fontSize=9,
                    horizontalalignment='left',
                    verticalalignment='bottom')

    plt.show()

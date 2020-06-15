import numpy as np
import pandas as pd
from sklearn import neighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

training_data = {
                 'NbStreams': [1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6],
                 'CriticRating': [1, 2, 5, 1, 2, 3, 5, 4, 5, 7, 4, 6, 5, 6, 7],
                 'IsPopular': [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
                }

new_instances = {
                 'NbStreams': [2, 2, 5],
                 'CriticRating': [2, 3, 7]
                }

df_training = pd.DataFrame(training_data, 
                           columns = ['NbStreams', 'CriticRating', 'IsPopular'])
df_new = pd.DataFrame(new_instances, 
                      columns = ['NbStreams', 'CriticRating'])

X_train = df_training[['NbStreams', 'CriticRating']].to_numpy()
y_train = df_training['IsPopular'].to_numpy().ravel()

X_new = df_new[['NbStreams', 'CriticRating']].to_numpy()


def plot_decision_boundaries(X_train, y_train, X_new, n_neighbors):

    h = .02

    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan'])
    cmap_bold = np.array(['darkorange', 'c'])

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    pred = clf.predict(X_new)
    
    # Plot the points

    myLabelMap = np.array(['isNotPopular', 'isPopular'])

    y_unique,id_unique = np.unique(y_train,return_index=True)
    X_unique = X_train[id_unique]
    X_train = np.asarray(X_train,dtype=float)

    for j,yj in enumerate(y_unique):
        plt.scatter(X_unique[j, 0], X_unique[j, 1], color=cmap_bold[yj],
                    edgecolor='k', s=20, label=myLabelMap[yj])

    X_train[id_unique] = np.nan
    plt.scatter(X_train[:, 0], X_train[:, 1], color=cmap_bold[y_train],
                edgecolor='k', s=20)

    plt.scatter(X_new[:, 0], X_new[:, 1], color=cmap_bold[pred], edgecolor='k', 
                s=20, marker='s')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('NbStreams (thousands)')
    plt.ylabel('CriticRating (0 to 10)')
    plt.legend()
    plt.title("IsPopular classification (k = {})".format(n_neighbors))

    plt.show()

plot_decision_boundaries(X_train, y_train, X_new, 1)
plot_decision_boundaries(X_train, y_train, X_new, 3)
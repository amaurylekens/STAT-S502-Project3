import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt



# build dataset
training_data = {
                 'Frozen_1':     [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                 'Has_Children': [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                 'Frozen_2':     [0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
                }

df_training = pd.DataFrame(training_data, 
                           columns = ['Frozen_1', 'Has_Children', 
                                      'Frozen_2'])

X_train = df_training[['Frozen_1', 'Has_Children']].to_numpy()
y_train = df_training['Frozen_2'].to_numpy().ravel()

# build decision tree
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)


# plot decision tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)

fn=['Frozen', 'Has_children']
cn=['yes', 'no']
plot_tree(clf,
          feature_names = fn, 
          class_names=cn,
          filled = True)

fig.savefig('decision_tree.png')

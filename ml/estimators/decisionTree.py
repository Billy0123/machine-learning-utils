from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

from ml.utils.plotDecisionRegions import plot_decision_regions
from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# import data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


# split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
iP.print(('Labels counts in y:', np.bincount(y)))
iP.print(('Labels counts in y_train:', np.bincount(y_train)))
iP.print(('Labels counts in y_test:', np.bincount(y_test)))


# decision tree run
tree = DecisionTreeClassifier(criterion='gini',
                              max_depth=4,
                              random_state=1)
tree.fit(X_train, y_train)


#plot results
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105, 150), title='Decision tree results')

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('decision tree results.png', dpi=300)
plt.show()


#draw decision tree
dot_data = export_graphviz(tree,
                           filled=True,
                           rounded=True,
                           class_names=['Setosa',
                                        'Versicolor',
                                        'Virginica'],
                           feature_names=['petal length',
                                          'petal width'],
                           out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('decision tree visualisation.png')


#random forest run
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)


#plot results
plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105, 150), title='Random forest results')

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('random forest results.png', dpi=300)
plt.show()

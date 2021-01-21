import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#svm stands for support vector machines they use hyper planes divides the data with anything straight
#soft margin allows few outliers in the margin
#hard margin does not allow any outliers in the margin
cancer = datasets.load_breast_cancer()
# print(cancer.feature_names)
# print(cancer.target_names)


x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.2)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear",C=2)
#clf = KNeighborsClassifier(n_neighbors=99)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
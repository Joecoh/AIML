from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR

# Classification
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Regression
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = SVR(kernel='rbf')
reg.fit(X_train, y_train)

from sklearn import datasets
wine = datasets.load_wine()

print("Features:", wine.feature_names)

print("Labels:", wine.target_names)

print("Data shape:", wine.data.shape)

print("First 5 feature records:\n", wine.data[0:5])

print("Labels:\n", wine.target)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn import metrics

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

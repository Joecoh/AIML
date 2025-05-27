from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

X, y = load_digits(return_X_y=True)
X = (X > 8).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = BernoulliNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(pred)

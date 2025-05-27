from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f'Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.2f}')
print(classification_report(y_test, rf_pred, target_names=iris.target_names))

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print(f'Gradient Boosting Accuracy: {accuracy_score(y_test, gb_pred):.2f}')
print(classification_report(y_test, gb_pred, target_names=iris.target_names))

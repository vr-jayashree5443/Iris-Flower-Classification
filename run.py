
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris_data = pd.read_csv("D:/JJ/Oasis Infobyte/1_Iris Flower Classification/archive (3)/Iris.csv")

label_to_species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

X = iris_data.drop(columns=["Species"])
y = iris_data["Species"]
#iris = datasets.load_iris()
#X = iris.data 
#y = iris.target  
target_names = ['setosa','versicolor','virginica']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
for row, species in zip(confusion, target_names):
    print(f"{species}: {row}")

print("\nPredicted Values   Actual Values")
for pred, actual in zip(y_pred, y_test):
    print(f"{pred}                {actual}")

print(f"\nAccuracy of the K-Nearest Neighbors classifier: {accuracy * 100:.2f}%")
confusion = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(8, 6))
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
for i in range(len(target_names)):
    for j in range(len(target_names)):
        plt.text(j, i, f'{confusion[i, j]}', ha='center', va='center', color='white')
plt.show()

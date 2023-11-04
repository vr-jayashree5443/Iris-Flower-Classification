# Iris-Flower-Classification
Let's walk through the workflow of the provided Python code step by step:

1. **Importing Libraries:** The code begins by importing the necessary libraries to perform tasks related to machine learning and data visualization. These libraries include Scikit-learn, Matplotlib, NumPy, and Pandas.

2. **Loading the Iris Dataset:**
   - The code loads the Iris dataset from a CSV file located at "D:/JJ/Oasis Infobyte/1_Iris Flower Classification/archive (3)/Iris.csv". This dataset contains measurements of iris flowers along with their corresponding species.

3. **Mapping Labels to Species Names:**
   - A mapping dictionary named `label_to_species` is created to associate numerical labels (0, 1, 2) with the corresponding species names ('setosa', 'versicolor', 'virginica'). This mapping is used to make the output more human-readable.

4. **Preparing Features and Target:**
   - The dataset is split into features (X) and the target variable (y). Features represent the measurements of the iris flowers, and the target variable is the species of each flower. The "Species" column is dropped to obtain the features.

5. **Splitting the Dataset:**
   - The dataset is divided into a training set and a test set. The training set contains 70% of the data, and the test set contains the remaining 30%. This split allows the model to learn from one portion and be evaluated on another. The random seed is set to 42 for reproducibility.

6. **Initializing and Training the KNN Classifier:**
   - A K-Nearest Neighbors (KNN) classifier is created with a parameter `n_neighbors=3`, indicating that it considers the three nearest neighbors when making predictions.
   - The KNN classifier is trained on the training data, which means it learns to recognize patterns in the feature measurements and their corresponding species labels.

7. **Making Predictions:**
   - The trained KNN classifier is used to make predictions on the test data. It calculates which species each iris flower in the test set most likely belongs to based on the measurements.

8. **Evaluating the Model:**
   - The code assesses the performance of the model:
     - It calculates the accuracy of the model by comparing the predicted species with the actual species from the test data. The accuracy score quantifies how well the model is at making correct predictions.
     - The code also computes and displays a confusion matrix, which provides a more detailed view of the model's performance, including true positives, true negatives, false positives, and false negatives for each species.

9. **Displaying Predicted and Actual Values:**
   - The code presents the predicted and actual species values side by side. It shows both numerical labels and species names, making it easier to understand the results.

10. **Plotting the Confusion Matrix:**
    - The confusion matrix is visualized using Matplotlib. The matrix is displayed as a grid where the color intensity in each cell represents the number of correct predictions for each species. Species names are used for labeling the axes, enhancing the readability of the confusion matrix.

In summary, this code performs the complete workflow for Iris flower classification, including data loading, model training, evaluation, and result visualization with a focus on making the output more interpretable by using species names instead of numerical labels.

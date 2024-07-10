import numpy as np
import openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Retrieve the data and extract the input features (X) and target (y)
dataset = openml.datasets.get_dataset('sick', download_data=True, download_qualities=True,
                                      download_features_meta_data=True)
X, y, _, feature_names = dataset.get_data(target='Class')
X = np.array(X)
y = y.to_numpy().ravel()

# Determine which features are string (discrete)
string_cols = dataset.get_features_by_type('string')
nominal_cols = dataset.get_features_by_type('nominal')
categorical_cols = list(set(string_cols + nominal_cols))
non_categorical_cols = list(set(range(len(X[0]))) - set(categorical_cols))

# Ensure categorical_cols indices are within valid range
max_index = X.shape[1] - 1
categorical_cols = [col for col in categorical_cols if col <= max_index]

# Print for debugging
print(f"Categorical columns: {categorical_cols}")
print(f"Non-categorical columns: {non_categorical_cols}")

# Encode categorical features using OneHotEncoder
onehot = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')
preproc = ColumnTransformer(transformers=[('onehot', onehot, categorical_cols)], remainder='passthrough')
X = preproc.fit_transform(X)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# If any features got mapped, they will have new names, rework the names to include original
onehot_names = preproc.get_feature_names_out()
for i in categorical_cols:
    onehot_names = [sub.replace(f'onehot__x{i}', feature_names[i]) for sub in onehot_names]
for i in non_categorical_cols:
    onehot_names = [sub.replace(f'remainder__x{i}', feature_names[i]) for sub in onehot_names]


#EXPERIMENT CODE HERE

#code for establishing parameters for models
hiddenLayerSizes = [2, 4, 6]
solvers = ['lbfgs', 'sgd', 'adam']
max_iterations = [500, 1000, 2000]
models = [Perceptron(shuffle=True,random_state=42,verbose=1,tol=0.0001)]


#code for creating models
for size in hiddenLayerSizes:
    for solver in solvers:
        for max_i in max_iterations:
            models.append(MLPClassifier(hidden_layer_sizes=(size),max_iter=max_i, solver=solver))


nfolds = 10
accuracies = np.zeros((len(models), nfolds))


kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)

# KFold cross validation with 10 folds
fold = 0
for train_index, test_index in kf.split(X):
    print(f"Fold {fold + 1}")

    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Code for fitting models
    for i, m in enumerate(models):
        m.fit(X_train, y_train.ravel())
        y_predict = m.predict(X_test)
        accuracies[i, fold] = accuracy_score(y_test, y_predict)

    fold += 1

# Get average accuracies
avg_accuracy = np.mean(accuracies, axis=1)

# Print average accuracies for each model
for i, accuracy in enumerate(avg_accuracy):
    print(f"Model {i + 1}: Average Accuracy = {accuracy}")


model_names = [f'Model {i+1}' for i in range(len(models))]
avg_accuracy = np.mean(accuracies, axis=1)

plt.figure(figsize=(10, 6))
plt.bar(model_names, avg_accuracy, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy of Models')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()



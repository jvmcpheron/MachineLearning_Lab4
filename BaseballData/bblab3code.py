import openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
# Retrieve the data and extract the input features (X) and target (y)
dataset = openml.datasets.get_dataset('baseball', download_data=True, download_qualities=True,
                                      download_features_meta_data=True)
X, y, _, feature_names = dataset.get_data(target='Hall_of_Fame')
X = np.array(X)
y = y.to_numpy().reshape(-1, 1)

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
Xnew = preproc.fit_transform(X)

# If any features got mapped, they will have new names, rework the names to include original
onehot_names = preproc.get_feature_names_out()
for i in categorical_cols:
    onehot_names = [sub.replace(f'onehot__x{i}', feature_names[i]) for sub in onehot_names]
for i in non_categorical_cols:
    onehot_names = [sub.replace(f'remainder__x{i}', feature_names[i]) for sub in onehot_names]

# Set the number of folds, repeats, and the different sizes for the training data used
nfolds = 10
nrepeats = 10
nsizes = 10
first_seed = 42


#DEFAULT DT
# Create the arrays to hold the accuracy and size for each size, repeat, fold combination
accuracies = np.zeros((nsizes, nrepeats, nfolds))
sizes = np.zeros((nsizes, nrepeats, nfolds))

# For each of a number of repeats
for r in range(nrepeats):
    # Divide the data using a stratified k-fold strategy
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=r + first_seed)

    # Keep track of the fold number
    f = 0

    # Get the indices of the overall data for each fold
    for train_index, test_index in skf.split(Xnew, y):
        # Extract the train and test data from the original Xnew and y
        Xtrain = Xnew[train_index, :]
        ytrain = y[train_index]
        Xtest = Xnew[test_index, :]
        ytest = y[test_index]

        # Determine the size of the train set
        num_examples = Xtrain.shape[0]

        # For each of the sizes (1/nsizes)*num_examples, (2/nsizes)*num_examples, ..., (nsizes/nsizes) * num_examples
        for z in range(nsizes):
            curr_size = round(((z + 1) / nsizes) * num_examples)

            # Use the first curr_size number of points from the train set
            Xtrainsub = Xtrain[0:curr_size, :]
            ytrainsub = ytrain[0:curr_size]

            # Learn a decision tree
            dtlearner = DecisionTreeClassifier()
            dtlearner.fit(Xtrainsub, ytrainsub)

            # Keep track of this size and calculate the accuracy for this size/repeat/fold combination
            sizes[z, r, f] = curr_size
            accuracies[z, r, f] = dtlearner.score(Xtest, ytest)
            print(f'Accuracy for size {curr_size}, repeat {r}, fold {f} is {accuracies[z, r, f]}')
        f += 1

# Create some matrices to store combined values
accuracies_across_folds = np.zeros((nsizes, nrepeats))
accuracies_by_size_dt = np.zeros((nsizes,))
std_by_size_dt = np.zeros((nsizes,))
average_size = np.zeros((nsizes,))

for z in range(nsizes):
    for r in range(nrepeats):
        # Combine the accuracy across all the folds for a size/repeat combination
        accuracies_across_folds[z][r] = np.mean(accuracies[z, r, :])

    # Calculate the mean and std of the accuracy for some size (calculating over repeats)
    accuracies_by_size_dt[z] = np.mean(accuracies_across_folds[z, :])
    std_by_size_dt[z] = np.std(accuracies_across_folds[z, :])

    # Calculate the average size
    average_size[z] = np.mean(sizes[z, :, :])
    print(f'For size {average_size[z]} avg accuracy is {accuracies_by_size_dt[z]} and std is {std_by_size_dt[z]}')



#DT ALT 2
# Create the arrays to hold the accuracy and size for each size, repeat, fold combination
accuracies = np.zeros((nsizes, nrepeats, nfolds))
sizes = np.zeros((nsizes, nrepeats, nfolds))

# For each of a number of repeats
for r in range(nrepeats):
    # Divide the data using a stratified k-fold strategy
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=r + first_seed)

    # Keep track of the fold number
    f = 0

    # Get the indices of the overall data for each fold
    for train_index, test_index in skf.split(Xnew, y):
        # Extract the train and test data from the original Xnew and y
        Xtrain = Xnew[train_index, :]
        ytrain = y[train_index]
        Xtest = Xnew[test_index, :]
        ytest = y[test_index]

        # Determine the size of the train set
        num_examples = Xtrain.shape[0]

        # For each of the sizes (1/nsizes)*num_examples, (2/nsizes)*num_examples, ..., (nsizes/nsizes) * num_examples
        for z in range(nsizes):
            curr_size = round(((z + 1) / nsizes) * num_examples)

            # Use the first curr_size number of points from the train set
            Xtrainsub = Xtrain[0:curr_size, :]
            ytrainsub = ytrain[0:curr_size]

            # Learn a decision tree
            dtlearner = DecisionTreeClassifier(min_samples_split=100)
            dtlearner.fit(Xtrainsub, ytrainsub)

            # Keep track of this size and calculate the accuracy for this size/repeat/fold combination
            sizes[z, r, f] = curr_size
            accuracies[z, r, f] = dtlearner.score(Xtest, ytest)
            print(f'Accuracy for size {curr_size}, repeat {r}, fold {f} is {accuracies[z, r, f]}')
        f += 1

# Create some matrices to store combined values
accuracies_across_folds = np.zeros((nsizes, nrepeats))
accuracies_by_size_dt2 = np.zeros((nsizes,))
std_by_size_dt2 = np.zeros((nsizes,))
average_size = np.zeros((nsizes,))

for z in range(nsizes):
    for r in range(nrepeats):
        # Combine the accuracy across all the folds for a size/repeat combination
        accuracies_across_folds[z][r] = np.mean(accuracies[z, r, :])

    # Calculate the mean and std of the accuracy for some size (calculating over repeats)
    accuracies_by_size_dt2[z] = np.mean(accuracies_across_folds[z, :])
    std_by_size_dt2[z] = np.std(accuracies_across_folds[z, :])

    # Calculate the average size
    average_size[z] = np.mean(sizes[z, :, :])
    print(f'For size {average_size[z]} avg accuracy is {accuracies_by_size_dt2[z]} and std is {std_by_size_dt2[z]}')


#DT ALT 3
# Create the arrays to hold the accuracy and size for each size, repeat, fold combination
accuracies = np.zeros((nsizes, nrepeats, nfolds))
sizes = np.zeros((nsizes, nrepeats, nfolds))

# For each of a number of repeats
for r in range(nrepeats):
    # Divide the data using a stratified k-fold strategy
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=r + first_seed)

    # Keep track of the fold number
    f = 0

    # Get the indices of the overall data for each fold
    for train_index, test_index in skf.split(Xnew, y):
        # Extract the train and test data from the original Xnew and y
        Xtrain = Xnew[train_index, :]
        ytrain = y[train_index]
        Xtest = Xnew[test_index, :]
        ytest = y[test_index]

        # Determine the size of the train set
        num_examples = Xtrain.shape[0]

        # For each of the sizes (1/nsizes)*num_examples, (2/nsizes)*num_examples, ..., (nsizes/nsizes) * num_examples
        for z in range(nsizes):
            curr_size = round(((z + 1) / nsizes) * num_examples)

            # Use the first curr_size number of points from the train set
            Xtrainsub = Xtrain[0:curr_size, :]
            ytrainsub = ytrain[0:curr_size]

            # Learn a decision tree
            dtlearner = DecisionTreeClassifier(min_samples_leaf=50)
            dtlearner.fit(Xtrainsub, ytrainsub)

            # Keep track of this size and calculate the accuracy for this size/repeat/fold combination
            sizes[z, r, f] = curr_size
            accuracies[z, r, f] = dtlearner.score(Xtest, ytest)
            print(f'Accuracy for size {curr_size}, repeat {r}, fold {f} is {accuracies[z, r, f]}')
        f += 1

# Create some matrices to store combined values
accuracies_across_folds = np.zeros((nsizes, nrepeats))
accuracies_by_size_dt3 = np.zeros((nsizes,))
std_by_size_dt3 = np.zeros((nsizes,))
average_size = np.zeros((nsizes,))

for z in range(nsizes):
    for r in range(nrepeats):
        # Combine the accuracy across all the folds for a size/repeat combination
        accuracies_across_folds[z][r] = np.mean(accuracies[z, r, :])

    # Calculate the mean and std of the accuracy for some size (calculating over repeats)
    accuracies_by_size_dt3[z] = np.mean(accuracies_across_folds[z, :])
    std_by_size_dt3[z] = np.std(accuracies_across_folds[z, :])

    # Calculate the average size
    average_size[z] = np.mean(sizes[z, :, :])
    print(f'For size {average_size[z]} avg accuracy is {accuracies_by_size_dt3[z]} and std is {std_by_size_dt3[z]}')


#NB
X, y, _, feature_names = dataset.get_data(target='Hall_of_Fame')
X = np.array(X)
y = y.to_numpy().ravel()

#Remove features
outfield_index = feature_names.index('Position')
X = np.delete(X, outfield_index, axis=1)

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

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Encode categorical features using OneHotEncoder
onehot = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')
preproc = ColumnTransformer(transformers=[('onehot', onehot, categorical_cols)], remainder='passthrough')
Xnew = preproc.fit_transform(X)

# If any features got mapped, they will have new names, rework the names to include original
onehot_names = preproc.get_feature_names_out()
for i in categorical_cols:
    onehot_names = [sub.replace(f'onehot__x{i}', feature_names[i]) for sub in onehot_names]
for i in non_categorical_cols:
    onehot_names = [sub.replace(f'remainder__x{i}', feature_names[i]) for sub in onehot_names]

# Set the number of folds, repeats, and the different sizes for the training data used
nfolds = 10
nrepeats = 10
nsizes = 10
first_seed = 42

# Create the arrays to hold the accuracy and size for each size, repeat, fold combination
accuracies = np.zeros((nsizes, nrepeats, nfolds))
sizes = np.zeros((nsizes, nrepeats, nfolds))

# For each of a number of repeats
for r in range(nrepeats):
    # Divide the data using a stratified k-fold strategy
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=r + first_seed)

    # Keep track of the fold number
    f = 0

    # Get the indices of the overall data for each fold
    for train_index, test_index in skf.split(Xnew, y):
        # Extract the train and test data from the original Xnew and y
        Xtrain = Xnew[train_index, :]
        ytrain = y[train_index]
        Xtest = Xnew[test_index, :]
        ytest = y[test_index]

        # Determine the size of the train set
        num_examples = Xtrain.shape[0]

        # For each of the sizes (1/nsizes)*num_examples, (2/nsizes)*num_examples, ..., (nsizes/nsizes) * num_examples
        for z in range(nsizes):
            curr_size = round(((z + 1) / nsizes) * num_examples)

            # Use the first curr_size number of points from the train set
            Xtrainsub = Xtrain[0:curr_size, :]
            ytrainsub = ytrain[0:curr_size]

            #Learn a Naive Bayes model
            nb_learner = GaussianNB()
            nb_learner.fit(Xtrainsub, ytrainsub)

            # Keep track of this size and calculate the accuracy for this size/repeat/fold combination
            sizes[z, r, f] = curr_size
            accuracies[z, r, f] = nb_learner.score(Xtest, ytest)
            print(f'Accuracy for size {curr_size}, repeat {r}, fold {f} is {accuracies[z, r, f]}')
        f += 1

# Create some matrices to store combined values
accuracies_across_folds = np.zeros((nsizes, nrepeats))
accuracies_by_size_nb = np.zeros((nsizes,))
std_by_size_nb = np.zeros((nsizes,))
average_size = np.zeros((nsizes,))

for z in range(nsizes):
    for r in range(nrepeats):
        # Combine the accuracy across all the folds for a size/repeat combination
        accuracies_across_folds[z][r] = np.mean(accuracies[z, r, :])

    # Calculate the mean and std of the accuracy for some size (calculating over repeats)
    accuracies_by_size_nb[z] = np.mean(accuracies_across_folds[z, :])
    std_by_size_nb[z] = np.std(accuracies_across_folds[z, :])

    # Calculate the average size
    average_size[z] = np.mean(sizes[z, :, :])
    print(f'For size {average_size[z]} avg accuracy is {accuracies_by_size_nb[z]} and std is {std_by_size_nb[z]}')




#PLOT HERE
# Plot the average accuracy showing the standard deviation as an error bar by the size of the train set
fig = plt.figure()

# Plot Decision Tree accuracy with green color
plt.errorbar(average_size, accuracies_by_size_dt, std_by_size_dt, color='green', label='Default Tree')
plt.errorbar(average_size, accuracies_by_size_dt2, std_by_size_dt2, color='blue', label='Split Edit Tree')
plt.errorbar(average_size, accuracies_by_size_dt3, std_by_size_dt3, color='purple', label='Leaf Edit Tree')
plt.errorbar(average_size, accuracies_by_size_nb, std_by_size_nb, color='red', label='Naive Bayes')


plt.xlabel("Size of training data")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
import numpy as np
import openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.impute import SimpleImputer

# Retrieve the data and extract the input features (X) and target (y)
dataset = openml.datasets.get_dataset('baseball', download_data=True, download_qualities=True,
                                      download_features_meta_data=True)
X, y, _, feature_names = dataset.get_data(target='Hall_of_Fame')
X = np.array(X)
y = y.to_numpy().reshape(-1, 1)

outfield_index = feature_names.index('Position')
X = np.delete(X, outfield_index, axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Determine which features are string (discrete)
string_cols = dataset.get_features_by_type('string')
nominal_cols = dataset.get_features_by_type('nominal')
categorical_cols = list(set(string_cols + nominal_cols))
non_categorical_cols = list(set(range(len(X[0]))) - set(categorical_cols))

# Ensure categorical_cols indices are within valid range
max_index = X.shape[1] - 1
categorical_cols = [col for col in categorical_cols if col <= max_index]

# Encode categorical features using OneHotEncoder
onehot = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')
preproc = ColumnTransformer(transformers=[('onehot', onehot, categorical_cols)], remainder='passthrough')
X = preproc.fit_transform(X)

# If any features got mapped, they will have new names, rework the names to include original
onehot_names = preproc.get_feature_names_out()
for i in categorical_cols:
    onehot_names = [sub.replace(f'onehot__x{i}', feature_names[i]) for sub in onehot_names]
for i in non_categorical_cols:
    onehot_names = [sub.replace(f'remainder__x{i}', feature_names[i]) for sub in onehot_names]


#fitting models
clf = Perceptron(shuffle=True,random_state=42,verbose=1,tol=0.0001)
clf.fit(X,y)

clf = MLPClassifier(hidden_layer_sizes=(2,),max_iter=5000,random_state=44,verbose=2,tol=0.0000001)
clf.fit(X,y)


#it appears we have working models
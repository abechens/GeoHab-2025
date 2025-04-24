import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import pickle 
#Split: 
with open('k_fold_investigate_errors.pkl', 'rb') as f:
    loaded_r2_scores_dict = pickle.load(f)
    loaded_grain_pred = pickle.load(f)

train_set_whole = pd.read_csv('data/train_set_with_feat_cleanCorr.csv')
train_set = train_set_whole


med_grain_pred = {keys: np.nanmedian(loaded_grain_pred[keys],axis=0) for keys in loaded_grain_pred.keys()}


diff_grain_size = {keys: train_set['mean_gs']-med_grain_pred[keys] for keys in  loaded_grain_pred.keys()}

key_to_watch = 20
error_thres = 5
idx_error = np.abs(diff_grain_size[key_to_watch])>=error_thres # True where we have outliers

idx_Corr = ~idx_error 
y = np.zeros(len(train_set))
y[idx_Corr] = 1
items_to_remove = ['id', 'mean_gs', 'sd', 'skewness', 'kurtosis', 'current_max', 'current_min','sample_type','y_im', 'x_im',  'x', 'y','comb_dist']

X = train_set.drop(items_to_remove, axis=1)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Apply SMOTE to handle the imbalanced dataset
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_smote, y_train_smote = X_train, y_train 
# Define the classifiers to be tested
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Balanced Random Forest": BalancedRandomForestClassifier()
}

# Dictionary to store the F1 scores of each classifier
penalized_f1_scores = {}
conf_mat = {}
# Iterate over classifiers and calculate F1 score using cross-validation
for name, clf in classifiers.items():
    clf.fit(X_train_smote, y_train_smote)
    y_pred = clf.predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)


    penalized_f1 = balanced_accuracy

   
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    penalized_f1_scores[name] = tn

# Rank classifiers based on F1 score
ranked_classifiers = sorted(penalized_f1_scores.items(), key=lambda item: item[1], reverse=True)

# Print the ranked classifiers and their F1 scores
for name, score in ranked_classifiers:
    print(f"{name}: {score:.4f}")

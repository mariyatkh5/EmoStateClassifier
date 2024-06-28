import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Define classifiers to evaluate
classifiers = {
    'KNN': KNeighborsClassifier(),
    'GLMNET': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, class_weight='balanced'),
    'SVML': SVC(kernel='linear', probability=True, class_weight='balanced'),
    'SVMR': SVC(kernel='rbf', probability=True, class_weight='balanced'),
    'DT': DecisionTreeClassifier(random_state=0, class_weight='balanced'),
    'NB': GaussianNB(),
    'XGBTREE': GradientBoostingClassifier(random_state=0),
    'LDA': LinearDiscriminantAnalysis(),
    'TBAG': BaggingClassifier(DecisionTreeClassifier(random_state=0, class_weight='balanced'), random_state=0)
}

# Mock data for demonstration, replace with actual data
np.random.seed(0)
X = np.random.rand(100, 6)  # Replace with actual data
y = np.random.randint(0, 2, 100)  # Replace with actual labels

# Extract the final subset of features for each speed
selected_features_speed1 = ['SCR_Peaks_N', 'SCR_Peaks_Amplitude_Mean', 'EDA_Tonic_SD']
selected_features_speed2 = ['SCR_Peaks_N', 'SCR_Peaks_Amplitude_Mean', 'EDA_SympatheticN']

# Mock data feature selection, replace with actual feature selection
X_speed1 = X[:, [0, 1, 2]]  # Replace with actual columns based on selected_features_speed1
X_speed2 = X[:, [0, 1, 4]]  # Replace with actual columns based on selected_features_speed2

# Prepare the data for box plots including all classifiers
def prepare_boxplot_data_all_classifiers(X, y, speed_label):
    data = []
    for clf_name, model in classifiers.items():
        skf = StratifiedKFold(n_splits=10)
        accuracy_scores = []
        auc_scores = []
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            accuracy = accuracy_score(y_test, y_pred)
            
            accuracy_scores.append(accuracy)
            auc_scores.append(roc_auc)
        
        for acc in accuracy_scores:
            if acc > 0:
                data.append({'Metric': 'Accuracy', 'Value': acc, 'Classifier': clf_name, 'Speed': speed_label})
        for auc_score in auc_scores:
            if auc_score > 0:
                data.append({'Metric': 'AUC', 'Value': auc_score, 'Classifier': clf_name, 'Speed': speed_label})
    
    return pd.DataFrame(data)

# Preparing the data for speed 1 and speed 2
data_speed1 = prepare_boxplot_data_all_classifiers(X_speed1, y, 'Speed Slow')
data_speed2 = prepare_boxplot_data_all_classifiers(X_speed2, y, 'Speed Fast')

# Plotting box plots for Speed 1
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Classifier', y='Value', hue='Metric', data=data_speed1)
plt.title('Box Plots for Accuracy and AUC - Robot Speed Slow')
plt.xticks(rotation=45)

# Plotting box plots for Speed 2
plt.subplot(1, 2, 2)
sns.boxplot(x='Classifier', y='Value', hue='Metric', data=data_speed2)
plt.title('Box Plots for Accuracy and AUC - Robot Speed Fast')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

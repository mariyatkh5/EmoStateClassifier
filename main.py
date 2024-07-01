import os
import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector
from deap import base, creator, tools, algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

# Suppress warnings and logging
warnings.filterwarnings("ignore", category=Warning)
logging.getLogger('sklearn').setLevel(logging.WARNING)

# Define classifiers to evaluate
classifiers = {
    'KNN': KNeighborsClassifier(),
    'GLMNET': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, class_weight='balanced'),
    'SVML': SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42),
    'SVMR': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    'DT': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'NB': GaussianNB(),
    'XGBTREE': GradientBoostingClassifier(random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'TBAG': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42, class_weight='balanced'), random_state=42),
    
}

# Iterate over each speed setting
for speed in [1, 2]:
    combined_data = []
    
    # Load data for each person and robot configuration
    for person in range(1, 26):
        for robots in [1, 2, 3]:
            file_path = f"analysis_results/Person_{person}/Analysis_{speed}_{robots}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                required_columns = [
                    "SCR_Peaks_N", "SCR_Peaks_Amplitude_Mean", "EDA_Tonic_SD",
                    "EDA_Sympathetic", "EDA_SympatheticN", "EDA_Autocorrelation",
                    f"{speed}.{robots}.3", f"{speed}.{robots}.4"
                ]
                
                # Check if required columns are present
                if all(col in df.columns for col in required_columns):
                    valence_value = df[f"{speed}.{robots}.3"].iloc[0]
                    arousal_value = df[f"{speed}.{robots}.4"].iloc[0]

                    # Define emotional states based on valence and arousal values
                    if 0 <= arousal_value <= 4 and 0 <= valence_value <= 6:
                        df["Emotional_State"] = 'Excitement'
                    elif 5 <= arousal_value <= 9 and 0 <= valence_value <= 6:
                        df["Emotional_State"] = 'Contentment'
                    else:
                        continue
                    
                    combined_data.append(df)
                else:
                    print(f"Required columns not found in {file_path}")

    if combined_data:
        # Combine data from all files
        combined_df = pd.concat(combined_data, ignore_index=True)

        eda_columns = ["SCR_Peaks_N", "SCR_Peaks_Amplitude_Mean", "EDA_Tonic_SD",
                       "EDA_Sympathetic", "EDA_SympatheticN", "EDA_Autocorrelation"]
        X = combined_df[eda_columns]
        y = combined_df["Emotional_State"].map({'Excitement': 1, 'Contentment': 0})

        print(f"Class distribution for speed {speed}:")
        print(y.value_counts())

        # Remove classes with less than 2 instances
        class_counts = y.value_counts()
        classes_to_keep = class_counts[class_counts >= 2].index
        mask = y.isin(classes_to_keep)
        X = X[mask]
        y = y[mask]

        if y.nunique() > 1:
            # Compute class weights for handling class imbalance
            class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

            # Define base pipeline components
            base_pipeline_steps = [
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]

            # Function to evaluate and store results
            results = []

            def evaluate_pipeline(method_name, model, X, y, classifier_name):
                # Perform cross-validation and store results
                kf = StratifiedKFold(n_splits=10)
                accuracies = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
                aucs = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
                
                results.append({
                    'Method': method_name,
                    'Classifier': classifier_name,
                    'Min Accuracy': accuracies.min(),
                    'Mean Accuracy': accuracies.mean(),
                    'Max Accuracy': accuracies.max(),
                    'Min AUC': aucs.min(),
                    'Mean AUC': aucs.mean(),
                    'Max AUC': aucs.max()
                })

                # Split data into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                model.fit(X_train, y_train)
                
                # Predictions and reports for training data
                y_pred_train = model.predict(X_train)
                report_train = classification_report(y_train, y_pred_train, target_names=['Contentment', 'Excitement'])
                print(f"\nClassification Report for {method_name} using {classifier_name} with speed {speed} (Training Data):\n")
                print(report_train)

                # Predictions and reports for test data
                y_pred_test = model.predict(X_test)
                report_test = classification_report(y_test, y_pred_test, target_names=['Contentment', 'Excitement'])
                print(f"\nClassification Report for {method_name} using {classifier_name} with speed {speed} (Test Data):\n")
                print(report_test)

                # Calculate confusion matrices
                cm_train = confusion_matrix(y_train, y_pred_train)
                cm_test = confusion_matrix(y_test, y_pred_test)
                print(f"\nConfusion Matrix for {method_name} using {classifier_name} with speed {speed} (Training Data):\n")
                print(cm_train)
                print(f"\nConfusion Matrix for {method_name} using {classifier_name} with speed {speed} (Test Data):\n")
                print(cm_test)

            # Feature selection using RF-RFE
            rf_rfe = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
            selector_rf_rfe = RFECV(estimator=rf_rfe, cv=StratifiedKFold(5), scoring='accuracy')
            selector_rf_rfe.fit(X, y)
            X_rf_rfe = selector_rf_rfe.transform(X)
            selected_features_rf_rfe = X.columns[selector_rf_rfe.get_support()].tolist()

            print("Selected Features after RF-RFE:")
            print(selected_features_rf_rfe)

            # Evaluate with RF-RFE selected features
            evaluate_pipeline("RF-RFE", RandomForestClassifier(random_state=42, class_weight=class_weight_dict), X_rf_rfe, y, "RandomForest")

            # Sequential Forward Selection (Stepwise Regression)
            sfs = SequentialFeatureSelector(LogisticRegression(max_iter=1000), k_features='parsimonious', forward=True, floating=False, scoring='accuracy', cv=StratifiedKFold(5))

            # Handle missing values for Sequential Feature Selector
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)

            sfs.fit(X_imputed, y)
            X_sfs = sfs.transform(X_imputed)
            selected_features_sfs = X.columns[list(sfs.k_feature_idx_)].tolist()

            print("Selected Features after Stepwise Forward Selection:")
            print(selected_features_sfs)

            # Evaluate with Stepwise Forward Selection selected features
            evaluate_pipeline("Stepwise Forward Selection", LogisticRegression(max_iter=1000, class_weight=class_weight_dict), X_sfs, y, "LogisticRegression")

            # Sequential Bidirectional Selection (Stepwise Bidirectional Regression)
            sbs = SequentialFeatureSelector(LogisticRegression(max_iter=1000), k_features='parsimonious', forward=False, floating=True, scoring='accuracy', cv=StratifiedKFold(5))

            sbs.fit(X_imputed, y)
            X_sbs = sbs.transform(X_imputed)
            selected_features_sbs = X.columns[list(sbs.k_feature_idx_)].tolist()

            print("Selected Features after Stepwise Bidirectional Selection:")
            print(selected_features_sbs)

            # Evaluate with Stepwise Bidirectional Selection selected features
            evaluate_pipeline("Stepwise Bidirectional Selection", LogisticRegression(max_iter=1000, class_weight=class_weight_dict), X_sbs, y, "LogisticRegression")

            # Genetic Algorithms (DEAP)
            def evaluate_individual(individual):
                # Evaluate individual feature subset
                features = [idx for idx, included in enumerate(individual) if included]
                if not features:
                    return 0,
                X_selected = X_imputed[:, features]
                clf = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
                kf = StratifiedKFold(n_splits=10)
                accuracies = cross_val_score(clf, X_selected, y, cv=kf, scoring='accuracy')
                return accuracies.mean(),

            # Setup DEAP for Genetic Algorithms
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()
            toolbox.register("attr_bool", np.random.randint, 0, 2)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(eda_columns))
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
            toolbox.register("select", tools.selTournament, tournsize=3)
            toolbox.register("evaluate", evaluate_individual)

            # Run Genetic Algorithms
            population = toolbox.population(n=50)
            ngen = 10
            cxpb = 0.5
            mutpb = 0.2

            result, _ = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)
            best_individual = tools.selBest(result, k=1)[0]
            selected_features_ga = [feature for feature, included in zip(eda_columns, best_individual) if included]

            print("Selected Features after Genetic Algorithms:")
            print(selected_features_ga)

            X_ga = X_imputed[:, [i for i, included in enumerate(best_individual) if included]]

            # Evaluate with Genetic Algorithms selected features
            evaluate_pipeline("Genetic Algorithms (DEAP)", 
                              RandomForestClassifier(random_state=42, class_weight=class_weight_dict), 
                              X_ga, y, "RandomForest")

            # Evaluate each classifier on different feature selection methods
            for method_name, feature_selector in zip(
                ["RF-RFE", "Stepwise Forward Selection", "Stepwise Bidirectional Selection", "Genetic Algorithms (DEAP)"],
                [selector_rf_rfe, sfs, sbs, best_individual]
            ):
                if method_name == "Genetic Algorithms (DEAP)":
                    X_method = X_imputed[:, [i for i, included in enumerate(feature_selector) if included]]
                else:
                    X_method = feature_selector.transform(X_imputed)

                print(f"Evaluating with {method_name} selected features:")
                for clf_name, clf in classifiers.items():
                    evaluate_pipeline(method_name, clf, X_method, y, clf_name)

            # Summarize results
            summary = []
            for method in ["RF-RFE", "Stepwise Forward Selection", 
                           "Stepwise Bidirectional Selection", "Genetic Algorithms (DEAP)"]:
                method_results = [r for r in results if r['Method'] == method]
                for clf in classifiers.keys():
                    clf_results = [r for r in method_results if r['Classifier'] == clf]
                    if clf_results:
                        summary.append({
                            'Method': method,
                            'Classifier': clf,
                            'Min Accuracy': min(r['Min Accuracy'] for r in clf_results),
                            'Mean Accuracy': np.mean([r['Mean Accuracy'] for r in clf_results]),
                            'Max Accuracy': max(r['Max Accuracy'] for r in clf_results),
                            'Min AUC': min(r['Min AUC'] for r in clf_results),
                            'Mean AUC': np.mean([r['Mean AUC'] for r in clf_results]),
                            'Max AUC': max(r['Max AUC'] for r in clf_results)
                        })

            summary_df = pd.DataFrame(summary)
            output_file = f'evaluation_results_ohensf{speed}.csv'
            summary_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

        else:
            print(f"Not enough classes to train the model for speed {speed}.")
    else:
        print(f"No valid data found for speed {speed}.")

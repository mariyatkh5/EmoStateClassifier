import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(speed):
    combined_data = []
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

                if all(col in df.columns for col in required_columns):
                    valence_value = df[f"{speed}.{robots}.3"].iloc[0]
                    arousal_value = df[f"{speed}.{robots}.4"].iloc[0]

                    if 0 <= arousal_value <= 4 and 0 <= valence_value <= 6:
                        df["Emotional_State"] = 'Excitement'
                    elif 5 <= arousal_value <= 9 and 0 <= valence_value <= 6:
                        df["Emotional_State"] = 'Contentment'
                    else:
                        continue

                    combined_data.append(df[required_columns + ["Emotional_State"]])

    return pd.concat(combined_data, ignore_index=True) if combined_data else pd.DataFrame()

# Perform one-way ANOVA for each predictor
def perform_anova(data):
    anova_results = []
    
    # Rename columns to be valid identifiers
    data = data.rename(columns=lambda x: x.replace('.', '_'))
    
    for column in ["SCR_Peaks_N", "SCR_Peaks_Amplitude_Mean", "EDA_Tonic_SD",
                   "EDA_Sympathetic", "EDA_SympatheticN", "EDA_Autocorrelation"]:
        if column in data.columns:
            model = ols(f'{column} ~ C(Emotional_State)', data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_value = anova_table["PR(>F)"][0]
            anova_results.append({
                'Feature': column,
                'F-Value': anova_table["F"][0],
                'P-Value': p_value
            })
    return pd.DataFrame(anova_results)

# Load, prepare data, and perform ANOVA for each speed
for speed in [1, 2]:
    data = load_and_prepare_data(speed)
    if not data.empty:
        anova_results = perform_anova(data)
        significant_features = anova_results[anova_results['P-Value'] < 0.05]

        # Save results to a CSV file
        anova_results.to_csv(f'anova_results_speed_{speed}.csv', index=False)
        significant_features.to_csv(f'significant_features_speed_{speed}.csv', index=False)

        # Display the results
        print(f"ANOVA Results for Speed {speed}:")
        print(anova_results)
        print(f"\nSignificant Features for Speed {speed}:")
        print(significant_features)

        # Optionally, you can visualize the distribution of significant features
        for feature in significant_features['Feature']:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Emotional_State', y=feature, data=data)
            plt.title(f'Box plot of {feature} by Emotional State for Speed {speed}')
            plt.show()
    else:
        print(f"No data available for Speed {speed}")
# Visualizing the distribution of each feature by emotional state
features = ["SCR_Peaks_N", "SCR_Peaks_Amplitude_Mean", "EDA_Tonic_SD",
            "EDA_Sympathetic", "EDA_SympatheticN", "EDA_Autocorrelation"]

for speed in [1, 2]:
    data = load_and_prepare_data(speed)
    if not data.empty:
        data = data.rename(columns=lambda x: x.replace('.', '_'))
        
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Emotional_State', y=feature, data=data)
            plt.title(f'Box plot of {feature} by Emotional State for Speed {speed}')
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.violinplot(x='Emotional_State', y=feature, data=data)
            plt.title(f'Violin plot of {feature} by Emotional State for Speed {speed}')
            plt.show()
    else:
        print(f"No data available for Speed {speed}")

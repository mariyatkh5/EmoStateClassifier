import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Laden der Daten aus der Excel-Datei
dateipfad = 'Questionnaire_data1.xlsx'
df = pd.read_excel(dateipfad)

# Initialisierung der Listen für die Plot-Daten
data_points_group2 = [] # Gruppe 2


# Definition der Farben und Marker für die verschiedenen Gruppen
speed_color = {'1': 'pink', '2': 'green'}
robot_marker = {'1': 'x', '2': '^', '3': '.'}

legend_elements = [
    plt.scatter([], [], color='pink', marker='o', s=100, label='Slow: 5 Robots'),
    plt.scatter([], [], color='green', marker='o', s=100, label='Fast: 5 Robots'),

    plt.scatter([], [], color='pink', marker='^', s=100, label='Slow: 15 Robots'),
    plt.scatter([], [], color='green', marker='^', s=100, label='Fast: 15 Robots'),
    plt.scatter([], [], color='pink', marker='x', s=100, label='Slow: 1 Robot'),
    plt.scatter([], [], color='green', marker='x', s=100, label='Fast: 1 Robot'),
]

# Funktion zur Skalierung der Werte
def scale_valence(value):
    return 5 - value

def scale_arousal(value):
    return value

# Verarbeitung der Daten für Gruppe 2
df_group2 = df[df['Group'] == 2]
for col in df_group2.columns:
    if col.endswith('.3'):
        speed = col.split('.')[0]
        num_robots = col.split('.')[1]
        for val in df_group2[col]:
            scaled_valence = scale_valence(val)
            scaled_arousal = scale_arousal(df_group2[f"{speed}.{num_robots}.4"][df_group2[col] == val].values[0])
            data_points_group2.append((scaled_valence, scaled_arousal, speed, robot_marker[num_robots]))

# Zählen der Häufigkeit der Datenpunkte für Gruppe 2
point_counts_group2 = Counter(data_points_group2)

# Erstellen von zwei Subplots (1x2) in einem separaten Rahmen
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Titel der Plots
titles = ['Not modified', 'Modified']

for i, ax in enumerate(axs.flat):
    ax.set_title(titles[i], pad=30, loc='left', fontsize=20, fontweight='bold')

# Zweiter Plot für Gruppe 2
for (x, y, speed, marker), count in point_counts_group2.items():
    jitter_x = np.random.uniform(-0.2, 0.2)
    jitter_y = np.random.uniform(-0.2, 0.2)
    marker_size = 50 + 50 * count
    if sum(1 for v in point_counts_group2.values() if v == count) > 1:
        marker_size = 50 + 20 * count
    axs[0].scatter(x + jitter_x, y + jitter_y, color=speed_color[speed], marker=marker, s=marker_size, alpha=0.6)

# Anpassung der Achsen für den zweiten Plot
axs[0].spines['left'].set_position('zero')
axs[0].spines['left'].set_color('black')
axs[0].spines['left'].set_linewidth(1.2)
axs[0].spines['right'].set_color('none')
axs[0].spines['top'].set_color('none')
axs[0].spines['bottom'].set_position(('data', 5))
axs[0].set_ylim(-1, 11)
axs[0].set_xlim(-6, 6)
axs[0].set_yticks([])
axs[0].set_xticks([])

# Emotions-Quadranten hinzufügen
axs[0].text(-4.5, 8, 'Distress', fontsize=14, fontweight='bold', ha='center')
axs[0].text(4.5, 8, 'Excitement', fontsize=14, fontweight='bold', ha='center')
axs[0].text(-4.5, 2, 'Depression', fontsize=14, fontweight='bold', ha='center')
axs[0].text(4.5, 2, 'Contentment', fontsize=14, fontweight='bold', ha='center')
axs[0].text(5.2, 5.2, 'Pleasure', fontsize=14, ha='center')
axs[0].text(-5.2, 5.2, 'Displeasure', fontsize=14, ha='center')
axs[0].text(0.75, -0.5, 'Sleepy', fontsize=14, ha='center')

# Vierter Plot für Gruppe 2 (Modified)
for (x, y, speed, marker), count in point_counts_group2.items():
    jitter_x = np.random.uniform(-0.2, 0.2)
    jitter_y = np.random.uniform(-0.2, 0.2)
    marker_size = 50 + 50 * count
    if sum(1 for v in point_counts_group2.values() if v == count) > 1:
        marker_size = 50 + 20 * count
    axs[1].scatter(x + jitter_x, y + jitter_y, color=speed_color[speed], marker=marker, s=marker_size, alpha=0.6)

# Anpassung der Achsen für den vierten Plot
axs[1].spines['left'].set_position(('data', -1.5))
axs[1].spines['left'].set_color('black')
axs[1].spines['left'].set_linewidth(1.2)
axs[1].spines['right'].set_color('none')
axs[1].spines['top'].set_color('none')
axs[1].spines['bottom'].set_position(('data', 5))
axs[1].set_ylim(-1, 11)
axs[1].set_xlim(-6, 6)
axs[1].set_yticks([])
axs[1].set_xticks([])

# Emotions-Quadranten hinzufügen
axs[1].text(-4.5, 8, 'Distress', fontsize=14, fontweight='bold', ha='center')
axs[1].text(4.5, 8, 'Excitement', fontsize=14, fontweight='bold', ha='center')
axs[1].text(-4.5, 2, 'Depression', fontsize=14, fontweight='bold', ha='center')
axs[1].text(4.5, 2, 'Contentment', fontsize=14, fontweight='bold', ha='center')
axs[1].text(5.2, 5.2, 'Pleasure', fontsize=14, ha='center')
axs[1].text(-5, 5.2, 'Displeasure', fontsize=14, ha='center')
axs[1].text(-0.75, -0.5, 'Sleepy', fontsize=14, ha='center')

# Achsenbeschriftungen hinzufügen
axs[0].text(7.2, 5, 'Valence', fontsize=17, fontweight='bold', ha='center')
axs[0].text(0, 11.3, 'Arousal', fontsize=17, fontweight='bold', ha='center')
axs[0].text(1.1, 10.5, 'Very active', fontsize=14, ha='center')

axs[1].text(-1.5, 11.3, 'Arousal', fontsize=17,fontweight='bold', ha='center')
axs[1].text(-0.4, 10.5, 'Very active', fontsize=14, ha='center')
axs[0].legend(handles=legend_elements, loc='upper left', title='Legend')

# Anzeige der Plots
plt.tight_layout()
plt.show()

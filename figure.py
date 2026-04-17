import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

models = [
    "K-Neighbors Classifier",
    "Decision Tree",
    "Random Forest",
    "Feed-Forward Network",
    "Support Vector Machine",
    "Logistic Regression",
    "Deep Learning ANN",
    "Naive Bayes"
]

##### OLD METRICS #####
# f1_5 = [0.891,0.908,0.901,0.904,0.910,0.893,0.908,0.881]
# f1_loocv = [0.888,0.873,0.882,0.901,0.877,0.899,0.897,0.884]

# acc_5 = [0.893,0.910,0.904,0.906,0.912,0.895,0.910,0.884]
# acc_loocv = [0.888,0.873,0.882,0.901,0.877,0.899,0.897,0.884]

# auc_5 = [0.943,0.973,0.973,0.971,0.973,0.955,0.972,0.952]
# auc_loocv = [0.946,0.965,0.957,0.960,0.957,0.937,0.975,0.946]

##### NEW METRICS #####
f1_5 = [0.900,0.887,0.901,0.897,0.893,0.890,0.897,0.893]
f1_loocv = [0.883,0.885,0.895,0.904,0.891,0.895,0.893,0.897]

acc_5 = [0.891,0.889,0.904,0.900,0.895,0.893,0.900,0.895]
acc_loocv = [0.885,0.889,0.898,0.906,0.893,0.898,0.895,0.900]

auc_5 = [0.956,0.964,0.975,0.963,0.955,0.968,0.963,0.956]
auc_loocv = [0.951,0.957,0.966,0.958,0.932,0.961,0.956,0.950]


color_5 = "#f4a3a3"
color_loocv = "#b22222"
header_color = "#f6c7c7"

def add_table(ax, title, five, loocv):
    ax.axis("off")

    rows = [[m, f"{a:.3f}", f"{b:.3f}"] for m,a,b in zip(models,five,loocv)]

    table = ax.table(
        cellText=rows,
        colLabels=["Tested Models","5-Fold","LOOCV"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)   # increase table text
    table.scale(1,1.2)       # increase row height

    for (row,col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.5)

        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")

    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")

def add_bar(ax, title, five, loocv, ylabel, ymin=0.85, ymax=1.0):
    x = np.arange(len(models))
    w = 0.35

    ax.bar(x-w/2, five, width=w, color=color_5, label="5-Fold")
    ax.bar(x+w/2, loocv, width=w, color=color_loocv, label="Leave One Out")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)

    ax.set_ylim(ymin=85, ymax=1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.legend(frameon=False, fontsize=8)

fig = plt.figure(figsize=(12,10))
gs = GridSpec(3,2, width_ratios=[1.1,1.6], hspace=0.5)

add_table(fig.add_subplot(gs[0,0]), "A\nF1 Score", f1_5, f1_loocv)
add_bar(fig.add_subplot(gs[0,1]), "F1 Scores of Tested Models", f1_5, f1_loocv, "F1 Score")

add_table(fig.add_subplot(gs[1,0]), "B\nAccuracy", acc_5, acc_loocv)
add_bar(fig.add_subplot(gs[1,1]), "Accuracy of Tested Models", acc_5, acc_loocv, "Accuracy")

add_table(fig.add_subplot(gs[2,0]), "C\nAUC", auc_5, auc_loocv)
add_bar(fig.add_subplot(gs[2,1]), "AUC of Tested Models", auc_5, auc_loocv, "AUC")

plt.tight_layout()
plt.show()
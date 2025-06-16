import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, average_precision_score, make_scorer, precision_recall_curve
from sklearn.preprocessing import label_binarize, LabelEncoder
import shap
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer

def calculate_evaluation_metrics(model, y_test, y_pred, y_pred_proba):
    print("\n=== Classification Report ===\n", classification_report(y_test, y_pred))
    
    is_binary = len(np.unique(y_test)) == 2
    classes = np.arange(len(np.unique(y_test)))

    if is_binary:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        pr_auc = average_precision_score(y_test, y_pred_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average="macro")
        pr_auc = average_precision_score(y_test, y_pred_proba, average="macro")

    print("\n=== ROC AUC and PR AUC values ===")
    print(f"ROC AUC: {roc_auc}")
    print(f"PR AUC: {pr_auc}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if is_binary:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC AUC = {auc(fpr, tpr):.2f}')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_title("ROC Curve")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend()

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, color='darkorange', lw=2, label=f'PR AUC = {pr_auc:.2f}')
        ax2.set_title("Precision-Recall Curve")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend()
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_test_bin.shape[1]
        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_title("Multiclass ROC Curve (OvR)")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend()

        for i, color in zip(range(n_classes), colors):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, color=color, lw=2, label=f'Class {i} (AUC = {pr_auc:.2f})')
        ax2.set_title("Multiclass Precision-Recall Curve (OvR)")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend()

    plt.tight_layout()
    plt.show()
    
    
def display_shap_plot(model, X_train, X_test):
    shap_explainer = shap.Explainer(model.predict_proba, X_train)
    shap_values = shap_explainer(X_test)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap.summary_plot(shap_values[1], X_test, plot_type="bar")
    else:
        shap.summary_plot(shap_values, X_test, plot_type="bar")
    
def display_lime_plot(model, X_train, X_test, index):
    lime_explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=[str(c) for c in model.classes_],
        mode='classification'
    )

    exp = lime_explainer.explain_instance(
        data_row=X_test.iloc[index].values,
        predict_fn=model.predict_proba
    )

    exp.show_in_notebook(show_table=True, show_all=False)
    

def plot_class_probability_distributions(y_test, y_pred_proba):
    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    
    n_classes = y_pred_proba.shape[1]
    class_names = ['bench', 'starter', 'all-star']
    for i in range(n_classes):
        sns.kdeplot(
            y_pred_proba[y_test == i, i], label='True', fill=True, color='blue', ax=ax[i]
        )
        sns.kdeplot(
            y_pred_proba[y_test != i, i], label='False', fill=True, color='red', ax=ax[i]
        )
        ax[i].set_title(f"Probability distribution for class '{class_names[i]}'")
        ax[i].set_ylabel("Density")
        ax[i].set_xlim(0, 1)
        ax[i].legend()
        ax[i].grid(True)

    plt.xlabel("Predicted probability")
    plt.tight_layout()
    plt.show()
    

def get_selected_features(n):
    return ['Seasons_played',
             f'GS_1_{n}',
             f'MP_1_{n}',
             f'FG_1_{n}',
             f'FT_1_{n}',
             f'TRB_1_{n}',
             f'TOV_1_{n}',
             f'PTS_1_{n}',
             f'PER_1_{n}',
             f'USG%_1_{n}',
             f'DWS_1_{n}',
             f'WS_1_{n}',
             f'WS/48_1_{n}',
             f'BPM_1_{n}',
             f'AS_1_{n}',
             'Player_class_num']
#!/usr/bin/env python
# coding: utf-8

# # 1. Importe 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt

RANDOM_STATE = 42


# # 2. Datenvorverarbeitung, Cross-Validation & Metriken

# In[2]:


def evaluate_model_cv(X, y, model_pipeline, model_name, cv=None, scoring=None):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if scoring is None:
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = cross_validate(model_pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
    summary = {}
    for metric in scoring:
        mean = np.mean(results[f'test_{metric}'])
        std = np.std(results[f'test_{metric}'])
        summary[metric] = f"{mean:.3f} ± {std:.3f}"
    print(f"{model_name}:")
    for k, v in summary.items():
        print(f"{k}: {v}")
    return results, summary


# # 4. Laden der German-Credit-Daten

# In[4]:


german_df = pd.read_csv("german_credit.csv")  
for col in german_df.columns:
    if german_df[col].dtype == 'object':
        german_df[col] = german_df[col].astype('category').cat.codes


# # 5. Fehlende Werte prüfen

# In[5]:


print("Fehlende Werte im German Credit Datensatz:\n", german_df.isnull().sum())


# # 6. Explorative Analyse

# ## Zielvariable: Verteilung

# In[25]:


print("Verteilung der Zielvariable (CreditRisk):")
print(german_df["CreditRisk"].value_counts())
german_df["CreditRisk"].value_counts().plot(kind="bar", title="Verteilung der Zielvariable")

plt.show()


# ## Numerische Feature-Verteilungen

# In[7]:


print("Numerische Feature-Verteilungen:")
german_df.describe().T


# ## Korrelationen

# In[44]:


corr = german_df.corr()
plt.figure(figsize=(16,12))  # Größer für Übersichtlichkeit
sns.heatmap(corr, annot=False, cmap="coolwarm",
            yticklabels=True, xticklabels=True)

plt.title("Korrelationsmatrix German Credit", fontsize=20)      # Titel-Schriftgröße
plt.xticks(fontsize=16)   # x-Achse Schriftgröße
plt.yticks(fontsize=16)   # y-Achse Schriftgröße
plt.tight_layout()
plt.show()


# # 8. Pipelines aufbauen — German Credit (Logistische Regression & Random Forest)

# In[10]:


rf_g_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))
])

lr_g_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE))
])


# # 9. Modelle evaluieren — German Credit

# In[26]:


# Cross-validation results
results_rf_g, summary_rf_g = evaluate_model_cv(X_german, y_german, rf_g_pipe, "Random Forest")
results_lr_g, summary_lr_g = evaluate_model_cv(X_german, y_german, lr_g_pipe, "Logistische Regression")

# Holdout split
X_train, X_test, y_train, y_test = train_test_split(
    X_german, y_german, test_size=0.2, stratify=y_german, random_state=RANDOM_STATE
)

# Fit models on train set
rf_g_pipe.fit(X_train, y_train)
lr_g_pipe.fit(X_train, y_train)

# Confusion matrices
for model_pipe, name in zip([rf_g_pipe, lr_g_pipe], ["Random Forest", "Logistische Regression"]):
    y_pred = model_pipe.predict(X_test)
    print(f"\n{name} — Confusion Matrix (Holdout):")
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.title(f"{name} Confusion Matrix (Holdout)")
    plt.show()

# ROC curves and AUC
rf_probs = rf_g_pipe.predict_proba(X_test)[:, 1]
lr_probs = lr_g_pipe.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
auc_rf = auc(fpr_rf, tpr_rf)
auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(10,7))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})", linewidth=3)
plt.plot(fpr_lr, tpr_lr, label=f"Logistische Regression (AUC = {auc_lr:.3f})", linewidth=3)
plt.plot([0, 1], [0, 1], 'k--', label="Chance", linewidth=2)
plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)
plt.title("ROC-Kurven Vergleich — German Credit", fontsize=18)
plt.legend(loc="lower right", fontsize=14)
plt.grid(True)
plt.show()


# # 10. Koeffizienten der Logistischen Regression interpretieren — German Credit

# In[39]:


plt.figure(figsize=(10, 9))  
plt.barh(coef_df_sorted['Feature'], coef_df_sorted['Koeffizient'])
plt.xlabel("Koeffizient", fontsize=16)
plt.title("LR-Koeffizienten — German Credit", fontsize=18)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()


# # 11. Feature-Importances des Random Forest interpretieren — German Credit

# In[41]:


plt.figure(figsize=(9, 9))
plt.barh(rf_df_sorted['Feature'], rf_df_sorted['Importance'])
plt.xlabel("Importance", fontsize=16)  
plt.title("RF Feature Importances — German Credit", fontsize=18)  
plt.yticks(fontsize=14)  
plt.tight_layout()  
plt.show()


# # 12. Laden der Credit-Card-Default-Daten (Taiwan)

# In[30]:


cc_df = pd.read_csv("default-of-credit-card-clients-2.csv")
cc_df = cc_df.drop(index=0).copy()
if 'ID' in cc_df.columns: cc_df = cc_df.drop('ID', axis=1)
if 'Y' in cc_df.columns: cc_df = cc_df.rename(columns={'Y': 'target'})
cc_df = cc_df.apply(pd.to_numeric, errors='coerce')


# # 13. Fehlende Werte prüfen

# In[31]:


print("Fehlende Werte im Credit Card Datensatz:\n", cc_df.isnull().sum())


# # 14. Explorative Analyse — Taiwan Credit Datensatz

# ## Zielvariable: Verteilung

# In[28]:


print("Verteilung der Zielvariable (target):")
print(cc_df["target"].value_counts())
cc_df["target"].value_counts().plot(kind="bar", title="Verteilung der Zielvariable (target)")
plt.show()


# ## Numerische Feature-Verteilungen

# In[17]:


print("Numerische Feature-Verteilungen:")
cc_df.describe().T


# ## Korrelationen

# In[45]:


corr_taiwan = cc_df.corr()
plt.figure(figsize=(16,12))
sns.heatmap(corr_taiwan, annot=False, cmap="coolwarm", yticklabels=True, xticklabels=True)
plt.title("Korrelationsmatrix Taiwan Credit", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()


# # 15. Zielvariable und Features definieren — Taiwan Credit

# In[19]:


y_cc = cc_df['target'].astype(int)
X_cc = cc_df.drop(columns=['target'])
print("Zielvariable — Verteilung:", y_cc.value_counts())


# # 16. Pipelines aufbauen — Taiwan Credit (Logistische Regression & Random Forest)

# In[20]:


rf_c_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))
])

lr_c_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE))
])


# # 17. Modelle evaluieren — Taiwan Credit

# In[29]:


# Cross-validation results
results_rf_c, summary_rf_c = evaluate_model_cv(X_cc, y_cc, rf_c_pipe, "Random Forest")
results_lr_c, summary_lr_c = evaluate_model_cv(X_cc, y_cc, lr_c_pipe, "Logistische Regression")

# Holdout split
X_train, X_test, y_train, y_test = train_test_split(
    X_cc, y_cc, test_size=0.2, stratify=y_cc, random_state=RANDOM_STATE
)

# Fit models
rf_c_pipe.fit(X_train, y_train)
lr_c_pipe.fit(X_train, y_train)

# Confusion matrices
for model_pipe, name in zip([rf_c_pipe, lr_c_pipe], ["Random Forest", "Logistische Regression"]):
    y_pred = model_pipe.predict(X_test)
    print(f"\n{name} — Confusion Matrix (Holdout):")
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.title(f"{name} Confusion Matrix (Holdout)")
    plt.show()

# ROC curves and AUC
rf_probs = rf_c_pipe.predict_proba(X_test)[:, 1]
lr_probs = lr_c_pipe.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
auc_rf = auc(fpr_rf, tpr_rf)
auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(10,7))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})", linewidth=3)
plt.plot(fpr_lr, tpr_lr, label=f"Logistische Regression (AUC = {auc_lr:.3f})", linewidth=3)
plt.plot([0, 1], [0, 1], 'k--', label="Chance", linewidth=2)
plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)
plt.title("ROC-Kurven Vergleich — Taiwan Credit", fontsize=18)
plt.legend(loc="lower right", fontsize=14)
plt.grid(True)
plt.show()


# # 18. Koeffizienten der Logistischen Regression interpretieren — Taiwan Credit

# In[49]:


plt.figure(figsize=(9, 9))  
plt.barh(coef_df_cc_sorted['Feature'], coef_df_cc_sorted['Koeffizient'])
plt.xlabel("Koeffizient", fontsize=16)  
plt.title("LR-Koeffizienten — Taiwan Credit", fontsize=18)  
plt.yticks(fontsize=14)  
plt.tight_layout()  
plt.show()


# # 19. Feature-Importances des Random Forest interpretieren — Taiwan Credit

# In[51]:


plt.figure(figsize=(9, 9)) 
plt.barh(rf_df_cc_sorted['Feature'], rf_df_cc_sorted['Importance'])
plt.xlabel("Importance", fontsize=16)  
plt.title("RF Feature Importances — Taiwan Credit", fontsize=18)  
plt.yticks(fontsize=14)  
plt.tight_layout()  
plt.show()


# # 20. Zusammenfassungstabelle (Mittelwert ± Std) 

# In[53]:


plt.figure(figsize=(28, 12))  # Noch größer!
plt.title("Cross-Validated Kennzahlen (Mittelwert ± Std)", fontsize=26, pad=30)
plt.axis('off')
tbl = plt.table(cellText=summary_all.values,
                rowLabels=summary_all.index,
                colLabels=summary_all.columns,
                cellLoc='center',
                loc='center',
                bbox=[0.05, 0.1, 0.9, 0.8])  # Abstand zu Rand!
tbl.auto_set_font_size(False)
tbl.set_fontsize(22)
tbl.scale(2.2, 2.5)
plt.tight_layout()
plt.show()


# In[ ]:





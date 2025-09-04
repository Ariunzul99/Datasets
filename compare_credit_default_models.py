import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(X, y, model, model_name, dataset_name):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n{dataset_name} - {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {model_name} ({dataset_name})")
    plt.show()
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    cv_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    print(f"Cross-validated Accuracy (mean): {cv_acc.mean():.4f}")
    print(f"Cross-validated ROC AUC (mean): {cv_auc.mean():.4f}")
    # Feature Importance (Random Forest only)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances.sort_values(ascending=False).plot(kind='bar', figsize=(10,4))
        plt.title(f"Feature Importance: {model_name} ({dataset_name})")
        plt.tight_layout()
        plt.show()
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return acc, roc_auc, cv_acc.mean(), cv_auc.mean(), fpr, tpr

# ----------- German Credit Data -----------
german_df = pd.read_csv("german_credit.csv")
for col in german_df.columns:
    if german_df[col].dtype == 'object':
        german_df[col] = german_df[col].astype('category').cat.codes
y_german = (german_df["CreditRisk"] == 2).astype(int)
X_german = german_df.drop("CreditRisk", axis=1)

rf_g = RandomForestClassifier(random_state=42)
lr_g = LogisticRegression(max_iter=1000, random_state=42)

results = []
acc_rf_g, auc_rf_g, cv_acc_rf_g, cv_auc_rf_g, fpr_rf_g, tpr_rf_g = evaluate_model(X_german, y_german, rf_g, "Random Forest", "German Credit")
acc_lr_g, auc_lr_g, cv_acc_lr_g, cv_auc_lr_g, fpr_lr_g, tpr_lr_g = evaluate_model(X_german, y_german, lr_g, "Logistic Regression", "German Credit")
results.append(["German Credit", "Random Forest", acc_rf_g, auc_rf_g, cv_acc_rf_g, cv_auc_rf_g])
results.append(["German Credit", "Logistic Regression", acc_lr_g, auc_lr_g, cv_acc_lr_g, cv_auc_lr_g])

# ----------- Credit Card Default Data -----------
cc_df = pd.read_csv("default-of-credit-card-clients-2.csv")
target_candidates = [col for col in cc_df.columns if 'default' in col.lower()]
if len(target_candidates) == 0:
    raise ValueError("No target column found!")
target_col_cc = target_candidates[0]
y_cc = cc_df[target_col_cc]
X_cc = cc_df.drop([target_col_cc], axis=1)
for col in X_cc.columns:
    if not np.issubdtype(X_cc[col].dtype, np.number):
        X_cc[col] = X_cc[col].astype('category').cat.codes

rf_c = RandomForestClassifier(random_state=42)
lr_c = LogisticRegression(max_iter=1000, random_state=42)

acc_rf_c, auc_rf_c, cv_acc_rf_c, cv_auc_rf_c, fpr_rf_c, tpr_rf_c = evaluate_model(X_cc, y_cc, rf_c, "Random Forest", "Credit Card Default")
acc_lr_c, auc_lr_c, cv_acc_lr_c, cv_auc_lr_c, fpr_lr_c, tpr_lr_c = evaluate_model(X_cc, y_cc, lr_c, "Logistic Regression", "Credit Card Default")
results.append(["Credit Card Default", "Random Forest", acc_rf_c, auc_rf_c, cv_acc_rf_c, cv_auc_rf_c])
results.append(["Credit Card Default", "Logistic Regression", acc_lr_c, auc_lr_c, cv_acc_lr_c, cv_auc_lr_c])

# ----------- ROC Curves Comparison -----------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(fpr_rf_g, tpr_rf_g, label='Random Forest')
plt.plot(fpr_lr_g, tpr_lr_g, label='Logistische Regression')
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve - German Credit Data")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.subplot(1,2,2)
plt.plot(fpr_rf_c, tpr_rf_c, label='Random Forest')
plt.plot(fpr_lr_c, tpr_lr_c, label='Logistische Regression')
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve - Credit Card Default Data")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

# ----------- Save Results -----------
results_df = pd.DataFrame(results, columns=["Dataset", "Model", "Accuracy", "ROC_AUC", "CV_Accuracy", "CV_ROC_AUC"])
results_df.to_csv("model_comparison_results.csv", index=False)
print("\nResults saved to model_comparison_results.csv")
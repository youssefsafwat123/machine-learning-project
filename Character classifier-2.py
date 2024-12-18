import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize

# Load dataset (example: digits dataset)
digits = load_digits()
X, y = digits.data, digits.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=30)  # Reduce to 30 dimensions
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# SMOTE for handling class imbalance (if needed)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train a classifier (e.g., Random Forest)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# AUC for each class
y_test_bin = label_binarize(y_test, classes=np.unique(y))  # Convert to binary format for AUC calculation
roc_auc = roc_auc_score(y_test_bin, y_prob, average=None, multi_class='ovr')  # AUC per class
print("AUC for each class:")
for i, auc in enumerate(roc_auc):
    print(f"Class {i}: AUC = {auc:.4f}")

# AUC (Macro Average)
auc_macro = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
print(f"\nAUC (Macro Average): {auc_macro:.4f}")

# ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(y_prob.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')

    # Display AUC value on the plot
    x_pos = 0.5
    y_pos = 0.05 + (i * 0.05)
    plt.text(x_pos, y_pos, f'AUC = {roc_auc[i]:0.2f}', fontsize=12)

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for each class')
plt.legend(loc='lower right')
plt.show()


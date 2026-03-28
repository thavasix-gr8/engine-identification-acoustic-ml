from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from extract_features import build_dataset
import numpy as np
import pickle

print("=" * 40)
print("  Vehicle Sound Classifier - Training")
print("=" * 40)

X, y = build_dataset("dataset/")

print(f"\nDataset ready:")
print(f"  Total : {len(X)} samples")
for label in np.unique(y):
    print(f"  {label:12s}: {sum(y == label)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# SVM with better params for 9 classes
model = SVC(
    kernel='rbf',
    C=10,           # increased from 1
    gamma='scale',
    probability=True,
    class_weight='balanced'  # handles unequal class sizes
)
model.fit(X_train_scaled, y_train)

y_pred   = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred) * 100

# Cross validation score
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\n  Accuracy      : {accuracy:.1f}%")
print(f"  CV Score      : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel saved!")
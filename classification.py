import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classify_churn(df, target_col):
    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X, drop_first=True)  # <-- needs pandas
    y = df[target_col].apply(lambda x: 1 if x in ['Yes','Churn','1'] else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    feat_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return y_test, y_pred, metrics, feat_imp
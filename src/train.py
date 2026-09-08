"""
train.py
يدرّب موديل XGBoost على بيانات المعاملات، يقيّم أداءه، ويحفظه لاستخدامه لاحقًا.
"""

import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import load_data, preprocess_data, split_data


def train_model(X_train, y_train):
    """
    يدرّب موديل XGBoost.
    نستخدم scale_pos_weight لأن حالات الاحتيال قليلة جدًا مقارنة بالسليمة،
    وهذا يخلي الموديل يعطي اهتمام أكبر للحالات القليلة (الاحتيال) بدل ما يتجاهلها.
    """
    fraud_count = y_train.sum()
    normal_count = len(y_train) - fraud_count
    scale_pos_weight = normal_count / fraud_count

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """يطبع تقرير أداء الموديل: Precision, Recall, F1-score، ومصفوفة الالتباس."""
    y_pred = model.predict(X_test)

    print("\n=== تقرير الأداء ===")
    print(classification_report(y_test, y_pred, target_names=["سليمة", "احتيال"]))

    print("=== مصفوفة الالتباس (Confusion Matrix) ===")
    print(confusion_matrix(y_test, y_pred))


def save_model(model, path="models/fraud_model.pkl"):
    """يحفظ الموديل المدرّب كملف لاستخدامه لاحقًا في الواجهة."""
    joblib.dump(model, path)
    print(f"\nتم حفظ الموديل بنجاح في: {path}")


if __name__=="__main__":
    print("loading data...")
    df = load_data()

    print("preprocessing data...")
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("trainging modle... (may take a minute or more)")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)
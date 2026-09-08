"""
preprocess.py
يقوم بتحميل بيانات المعاملات، تنظيفها، وتقسيمها لتدريب/اختبار.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path="data/creditcard.csv"):
    """يقرأ ملف CSV ويرجعه كـ DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """
    يجهز البيانات للتدريب:
    - يوحّد مقياس عمود Amount (لأنه بمقياس مختلف عن باقي الأعمدة V1-V28)
    - يحذف عمود Time (لا يحمل معنى مفيد للتصنيف)
    - يفصل الخصائص (X) عن الهدف (y)
    """
    data = df.copy()

    scaler = StandardScaler()
    data["Amount"] = scaler.fit_transform(data[["Amount"]])

    if "Time" in data.columns:
        data = data.drop(columns=["Time"])

    X = data.drop(columns=["Class"])
    y = data["Class"]

    return X, y, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """يقسم البيانات إلى تدريب واختبار مع الحفاظ على نسبة الاحتيال بكلا الجزئين."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__=="__main__":
    df = load_data()
    print("عدد الصفوف والأعمدة:", df.shape)
    print("توزيع الفئات (0=سليمة, 1=احتيال):")
    print(df["Class"].value_counts())

    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nحجم بيانات التدريب:", X_train.shape)
    print("حجم بيانات الاختبار:", X_test.shape)
"""
explain.py
يستخدم SHAP لتفسير قرارات موديل XGBoost - يوضح أي الخصائص أثرت على كل قرار وبأي اتجاه.
"""

import shap
import joblib
import matplotlib.pyplot as plt

from preprocess import load_data, preprocess_data, split_data


def load_model(path="models/fraud_model.pkl"):
    """يحمّل الموديل المدرّب من الملف المحفوظ."""
    return joblib.load(path)


def create_explainer(model):
    """
    ينشئ SHAP Explainer مخصص لموديلات الأشجار (Tree-based) مثل XGBoost.
    TreeExplainer سريع جدًا ومناسب تحديدًا لهذا النوع من الموديلات، وهذا ما يمكّننا
    من حساب التفسير لحظيًا (Real-Time) بدون تأخير ملحوظ.
    """
    explainer = shap.TreeExplainer(model)
    return explainer


def explain_transaction(explainer, X_sample):
    """
    يحسب قيم SHAP لمعاملة واحدة (أو عدة معاملات).
    كل قيمة SHAP توضح مقدار تأثير كل خاصية (feature) على القرار النهائي:
    - قيمة موجبة = دفعت القرار نحو "احتيال"
    - قيمة سالبة = دفعت القرار نحو "سليمة"
    """
    shap_values = explainer.shap_values(X_sample)
    return shap_values


def plot_explanation(explainer, shap_values, X_sample, save_path=None):
    """يرسم رسم SHAP يوضح بصريًا تأثير كل خاصية على قرار معاملة واحدة."""
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_sample.iloc[0],
            feature_names=X_sample.columns.tolist()
        ),
        show=False
    )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


if __name__=="__main__":
    print("Loading model and data...")
    model = load_model()
    df = load_data()
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Creating SHAP explainer...")
    explainer = create_explainer(model)

    # نختار أول معاملة من بيانات الاختبار كمثال للتجربة
    sample = X_test.iloc[[0]]

    print("Computing SHAP values for a sample transaction...")
    shap_values = explain_transaction(explainer, sample)

    print("Plotting explanation...")
    plot_explanation(explainer, shap_values, sample, save_path="sample_explanation.png")
"""
app.py
واجهة Streamlit تفاعلية لعرض نظام كشف الاحتيال المالي مع تفسير SHAP لحظي (Real-Time).
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from preprocess import load_data, preprocess_data, split_data
from explain import load_model, create_explainer


st.set_page_config(page_title="كشف الاحتيال المالي", page_icon="🔍", layout="wide")


@st.cache_resource
def get_model_and_explainer():
    model = load_model()
    explainer = create_explainer(model)
    return model, explainer


@st.cache_data
def get_test_data():
    df = load_data()
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_test, y_test


def draw_sample(X_test, y_test, only_fraud):
    if only_fraud:
        fraud_indices = y_test[y_test == 1].index
        return X_test.loc[fraud_indices].sample(1).index[0]
    return X_test.sample(1).index[0]


def clamp(value, min_val, max_val):
    return min(max(value, min_val), max_val)


def main():
    st.title("🔍 نظام كشف الاحتيال المالي مع الذكاء الاصطناعي القابل للتفسير")
    st.markdown("يستخدم هذا النظام موديل **XGBoost** لتصنيف المعاملات، مع **SHAP** لتفسير كل قرار لحظيًا.")

    model, explainer = get_model_and_explainer()
    X_test, y_test = get_test_data()

    st.sidebar.header("⚙️ اختر طريقة الاختبار")
    mode = st.sidebar.radio("طريقة الإدخال:", ["معاملة عشوائية من البيانات", "إدخال يدوي"])

    if mode == "معاملة عشوائية من البيانات":
        only_fraud = st.sidebar.checkbox("🎯 اسحب فقط معاملات احتيالية (Fraud)", value=True)

        if st.sidebar.button("🎲 اسحب معاملة جديدة"):
            st.session_state["sample_index"] = draw_sample(X_test, y_test, only_fraud)

        if "sample_index" not in st.session_state:
            st.session_state["sample_index"] = draw_sample(X_test, y_test, only_fraud)

        sample = X_test.loc[[st.session_state["sample_index"]]]
        actual_label = y_test.loc[st.session_state["sample_index"]]

        st.subheader("📋 بيانات المعاملة المختارة")
        st.dataframe(sample)
        st.caption(f"الفئة الحقيقية لهذه المعاملة: {'احتيال' if actual_label == 1 else 'سليمة'}")

    else:
        st.subheader("✍️ أدخل قيم المعاملة يدويًا")
        st.caption("القيم **بعد التطبيع**. حرّك V14 لليسار (قيم سالبة كبيرة) لرفع احتمالية الاحتيال!")

        # ⭐ نسحب المعاملة الأساسية مرة واحدة فقط
        if "manual_base" not in st.session_state:
            fraud_indices = y_test[y_test == 1].index
            st.session_state["manual_base"] = X_test.loc[fraud_indices].sample(1).copy()

        base_sample = st.session_state["manual_base"]

        AMOUNT_MIN, AMOUNT_MAX = -2.0, 8.0
        V14_MIN, V14_MAX = -20.0, 5.0

        # ⭐ نهّيء قيم الشرائح بالجلسة (مرة واحدة فقط)
        if "slider_amount" not in st.session_state:
            st.session_state.slider_amount = clamp(float(base_sample["Amount"].values[0]), AMOUNT_MIN, AMOUNT_MAX)
        if "slider_v14" not in st.session_state:
            st.session_state.slider_v14 = clamp(float(base_sample["V14"].values[0]), V14_MIN, V14_MAX)

        # ⭐ الشرائح بـ keys — Streamlit يتذكر مكانها تلقائيًا وما يرجّعها
        col1, col2 = st.columns(2)
        with col1:
            amount_value = st.slider("قيمة Amount (بعد التطبيع)", AMOUNT_MIN, AMOUNT_MAX, key="slider_amount")
        with col2:
            v14_value = st.slider("قيمة V14 (الأقوى تأثيرًا)", V14_MIN, V14_MAX, key="slider_v14")

        # ⭐ نطبق القيم على نسخة من البيانات
        sample = base_sample.copy()
        sample["Amount"] = amount_value
        sample["V14"] = v14_value

        st.subheader("📋 بيانات المعاملة الحالية")
        st.dataframe(sample)

    if st.button("🔎 تحقق من المعاملة الآن", type="primary"):
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("⚠️ القرار: معاملة احتيالية")
            else:
                st.success("✅ القرار: معاملة سليمة")
            st.metric("نسبة احتمالية الاحتيال", f"{probability * 100:.2f}%")
            st.progress(min(float(probability), 1.0))

        with col2:
            st.info("النموذج يوضح أدناه أي الخصائص أثرت على هذا القرار وبأي اتجاه.")

        st.subheader("🧠 تفسير القرار (SHAP)")
        shap_values = explainer.shap_values(sample)

        base_values = explainer.expected_value
        if hasattr(base_values, "__len__"):
            base_values = base_values[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=base_values,
                data=sample.iloc[0],
                feature_names=sample.columns.tolist()
            ),
            show=False
        )
        st.pyplot(fig)

        st.caption("الأشرطة الحمراء (تدفع لليمين) تزيد احتمالية الاحتيال، والزرقاء (تدفع لليسار) تقلله.")


if __name__ == "__main__":
    main()
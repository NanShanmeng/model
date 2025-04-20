import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import xgboost as xgb

model = joblib.load('XGBoost.pkl')
X_test = pd.read_csv('X_test.csv')
feature_names = [
    "SLC6A13",
    "ANLN",
    "MARCO",
    "SYT13",
    "ARG2",
    "MEFV",
    "ZNF29P",
    "FLVCR2",
    "PTGFR",
    "CRISP2",
    "EME1",
    "IL22RA2",
    "SLC29A4",
    "CYBB",
    "LRRC25",
    "SCN8A",
    "LILRA6",
    "CTD_3080P12_3",
    "PECAM1"
]
st.title("Predicting the risk of non-small cell lung cancer based on the expression levels of diabetes-related genes")
SLC6A13 = st.number_input("SLC6A13", min_value=0, max_value=100000, value=161)
ANLN = st.number_input("ANLN", min_value=0, max_value=100000, value=161)
MARCO = st.number_input("MARCO", min_value=0, max_value=100000, value=1439)
SYT13 = st.number_input("SYT13", min_value=0, max_value=100000, value=12)
ARG2 = st.number_input("ARG2", min_value=0, max_value=100000, value=3224)
MEFV = st.number_input("MEFV", min_value=0, max_value=100000, value=34)
ZNF29P = st.number_input("ZNF29P", min_value=0, max_value=100000, value=1)
FLVCR2 = st.number_input("FLVCR2", min_value=0, max_value=100000, value=654)
PTGFR = st.number_input("PTGFR", min_value=0, max_value=100000, value=24)
CRISP2 = st.number_input("CRISP2", min_value=0, max_value=100000, value=44)
EME1 = st.number_input("EME1", min_value=0, max_value=100000, value=495)
IL22RA2 = st.number_input("IL22RA2", min_value=0, max_value=100000, value=12)
SLC29A4 = st.number_input("SLC29A4", min_value=0, max_value=100000, value=913)
CYBB = st.number_input("CYBB", min_value=0, max_value=100000, value=1629)
LRRC25 = st.number_input("LRRC25", min_value=0, max_value=100000, value=288)
SCN8A = st.number_input("SCN8A", min_value=0, max_value=100000, value=714)
LILRA6 = st.number_input("LILRA6", min_value=0, max_value=100000, value=113)
CTD_3080P12_3 = st.number_input("CTD_3080P12_3", min_value=0, max_value=100000, value=1)
PECAM1 = st.number_input("PECAM1", min_value=0, max_value=100000, value=5020)
feature_values = [SLC6A13, ANLN, MARCO, SYT13, ARG2, MEFV, ZNF29P, FLVCR2, PTGFR, CRISP2, EME1, IL22RA2,
                  SLC29A4, CYBB, LRRC25, SCN8A, LILRA6, CTD_3080P12_3, PECAM1]
feature = np.array([feature_values])
if st.button("Predict"):
    predicted_class = model.predict(feature)[0]
    predicted_proba = model.predict_proba(feature)[0]
    st.write(f"**Predicted Class:** {predicted_class} (1: Tumor, 0: Normal)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (f"According to our model, we're sorry to tell you that you're at high risk of having non - small cell lung cancer. Please contact a professional doctor for a thorough check - up as soon as possible. Note that our result isn't a final diagnosis. The specific result should be based on the diagnosis from a relevant hospital.")
    else:
        advice = (f"According to our model, we're glad to inform you that your risk of non - small cell lung cancer is low. But if you feel unwell, consult a professional doctor. Wish you good health. Note that our result isn't a final diagnosis. The specific result should be based on the diagnosis from a relevant hospital.")
    st.write(advice)
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
# 确保 explainer_shap.expected_value 和 shap_values 的结构正确
# 假设 explainer_shap 是 SHAP 解释器，shap_values 是计算得到的 SHAP 值
# sample_index 是你想要解释的样本索引
# X_test 是测试数据集，feature_names 是特征名称列表
import shap

# 假设 model 是你的训练好的模型，X_train 是训练数据
explainer_shap = shap.TreeExplainer(model)

# 计算 SHAP 值
shap_values = explainer_shap.shap_values(X_test)

# 确保 sample_index 是有效的
sample_index = 9
if 0 <= sample_index < len(X_test):
    # 如果模型是二分类模型，并且 expected_value 是一个数组，取第二个类别的 expected_value
    # 否则，直接使用 expected_value
    expected_value = explainer_shap.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
        expected_value = expected_value[1]

    # 从 shap_values 中获取对应样本的 SHAP 值
    # 如果 shap_values 是一个列表（例如，对于二分类模型），选择第二个类别的 SHAP 值
    sample_shap_values = shap_values[sample_index, :]
    if isinstance(shap_values, list) and len(shap_values) > 1:
        sample_shap_values = shap_values[1][sample_index, :]

    # 获取对应的特征值
    sample_features = X_test.iloc[sample_index, :]

    # 创建 DataFrame
    sample_df = pd.DataFrame([sample_features], columns=feature_names)

    # 生成 force_plot
    shap.force_plot(
        expected_value,
        sample_shap_values,
        sample_df,
        matplotlib=True,
        show=False
    )
else:
    print(f"None: {sample_index}")    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP FOrce Plot Explanation')
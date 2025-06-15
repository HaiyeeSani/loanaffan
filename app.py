import streamlit as st
import pandas as pd
import joblib
import json

# --- Page Configuration ---
# Set the configuration for the Streamlit page.
st.set_page_config(page_title="NPL Prediction App", page_icon="💸", layout="centered")

# --- Resource Loading Function ---
@st.cache_resource
def load_resources():
    """
    Load the trained model pipeline and UI options from disk.
    Using st.cache_resource to prevent reloading on every interaction.
    """
    # It's crucial that this model was trained on the 'อัตราส่วนชำระต่อรายได้' feature.
    model = joblib.load('pipeline.joblib')
    with open('options.json', 'r', encoding='utf8') as f:
        options = json.load(f)
    return model, options

# --- Load Model and Options ---
# Try to load the necessary files. If they don't exist, show an error and stop.
try:
    model, options = load_resources()
except FileNotFoundError:
    st.error("ไม่พบไฟล์โมเดล 'pipeline.joblib' หรือ 'options.json'. กรุณารันสคริปต์สำหรับฝึกโมเดลก่อน")
    st.stop()


# --- User Interface ---
st.title("💸 เครื่องมือประเมินความเสี่ยงสินเชื่อ (NPL)")
st.write("กรอกข้อมูลเพื่อประเมินความเสี่ยงของสมาชิกที่ขอสินเชื่อใหม่ โดยระบบจะแบ่งระดับความเสี่ยงเป็น 4 ระดับ")

# Create a form for a cleaner layout and single submission button.
with st.form("input_form"):
    st.header("ข้อมูลสมาชิก")

    # Arrange input fields into two columns for a better UI.
    col1, col2 = st.columns(2)
    
    with col1:
        # Categorical feature inputs using select boxes populated from options.json
        occupation = st.selectbox("อาชีพ (Occupation)", options=options['อาชีพ'])
        purpose = st.selectbox("วัตถุประสงค์ (Purpose)", options=options['วัตถุประสงค์'])
        status = st.selectbox("สถานภาพ (Marital Status)", options=options['สถานภาพ'])
        gender = st.selectbox("เพศ (Gender)", options=options['เพศ'])
    
    with col2:
        # Numerical feature inputs
        income = st.number_input("รายได้ (ต่อเดือน)", min_value=0, value=15000, step=1000)
        loan_amount = st.number_input("วงเงินที่ขอ", min_value=0, value=50000, step=5000)
        # --- EDITED: Changed DTI input to Monthly Payment input ---
        # Instead of asking for the ratio, we ask for the monthly payment to calculate it.
        monthly_payment = st.number_input("ชำระต่องวด (บาทต่อเดือน)", min_value=0, value=5000, step=500)


    # The submission button for the form.
    submitted = st.form_submit_button("ประเมินความเสี่ยง")

# --- Results Display Section ---
if submitted:
    # --- EDITED: Calculate DTI from inputs ---
    # The model expects the 'อัตราส่วนชำระต่อรายได้' feature.
    # We calculate it here from the user's inputs before prediction.
    # Added a check for income > 0 to avoid division by zero error.
    if income > 0:
        dti_calculated = (monthly_payment / income) * 100
    else:
        dti_calculated = 0 # Default to 0 if income is 0

    # Collect user input into a pandas DataFrame.
    input_data = pd.DataFrame({
        'อาชีพ': [occupation],
        'วัตถุประสงค์': [purpose],
        'สถานภาพ': [status],
        'เพศ': [gender],
        'รายได้': [income],
        'วงเงินที่ขอ': [loan_amount],
        # The model was trained with this feature name, so we must use it.
        'อัตราส่วนชำระต่อรายได้': [dti_calculated]
    })

    # Use the model to predict the probability of being an NPL.
    predict_proba = model.predict_proba(input_data)[0]
    npl_probability = predict_proba[1]

    st.header("🎯 ผลการประเมิน")

    # Define risk levels based on NPL probability.
    if npl_probability < 0.25: # 0% - 24.99%
        st.success(f"**เสี่ยงต่ำ** (ความน่าจะเป็น NPL: {npl_probability:.2%})", icon="✅")
        st.write("แนวโน้มการอนุมัติ: **สูง**")
    elif npl_probability < 0.50: # 25% - 49.99%
        st.info(f"**เสี่ยงปานกลาง** (ความน่าจะเป็น NPL: {npl_probability:.2%})", icon="🤔")
        st.write("แนวโน้มการอนุมัติ: **พิจารณาเพิ่มเติม** อาจต้องการข้อมูลหรือหลักประกันเพิ่ม")
    elif npl_probability < 0.75: # 50% - 74.99%
        st.warning(f"**เสี่ยงสูง** (ความน่าจะเป็น NPL: {npl_probability:.2%})", icon="⚠️")
        st.write("แนวโน้มการอนุมัติ: **ต่ำ** ควรมีหลักทรัพย์ค้ำประกันหรือผู้ค้ำประกันที่น่าเชื่อถือ")
    else: # 75% - 100%
        st.error(f"**เสี่ยงสูงมาก** (ความน่าจะเป็น NPL: {npl_probability:.2%})", icon="🚨")
        st.write("แนวโน้มการอนุมัติ: **ต่ำมาก** ไม่แนะนำให้อนุมัติสินเชื่อ")

    # Display a progress bar to visually represent the NPL probability.
    st.progress(npl_probability)
    st.caption("แถบด้านบนแสดงความน่าจะเป็นที่ลูกค้ารายนี้จะกลายเป็นหนี้เสีย (NPL)")

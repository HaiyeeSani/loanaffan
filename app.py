import streamlit as st
import pandas as pd
import joblib
import json
from dotenv import load_dotenv
import os
import requests # เพิ่มไลบรารี requests สำหรับเรียก API

# โหลด .env (ถ้ามี)
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="NPL Prediction App", page_icon="💸", layout="wide")

# --- โหลดโมเดลและ options ---
@st.cache_resource
def load_resources():
    """โหลดโมเดล Machine Learning และไฟล์ options."""
    try:
        model = joblib.load('pipeline.joblib')
        with open('options.json', 'r', encoding='utf8') as f:
            options = json.load(f)
        return model, options
    except FileNotFoundError:
        st.error("ไม่พบไฟล์โมเดล 'pipeline.joblib' หรือ 'options.json'. กรุณารันสคริปต์สำหรับฝึกโมเดลก่อน.")
        st.stop()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดทรัพยากร: {e}")
        st.stop()

model, options = load_resources()

# --- UI หลัก ---
st.title("💸 เครื่องมือประเมินความเสี่ยงและให้คำปรึกษาด้านสินเชื่อ")

col_app, col_info = st.columns([2,1])

with col_app:
    st.header("ประเมินความเสี่ยง NPL")
    st.write("กรอกข้อมูลเพื่อประเมินความเสี่ยงของสมาชิกที่ขอสินเชื่อใหม่")

    with st.form("input_form"):
        st.subheader("ข้อมูลสมาชิก")
        c1, c2 = st.columns(2)
        with c1:
            occupation = st.selectbox("อาชีพ (Occupation)", options=options['อาชีพ'])
            purpose = st.selectbox("วัตถุประสงค์ (Purpose)", options=options['วัตถุประสงค์'])
            status = st.selectbox("สถานภาพ (Marital Status)", options=options['สถานภาพ'])
            gender = st.selectbox("เพศ (Gender)", options=options['เพศ'])
        with c2:
            income = st.number_input("รายได้ (ต่อเดือน)", min_value=0, value=15000, step=500)
            loan_amount = st.number_input("วงเงินที่ขอ", min_value=0, value=50000, step=1000)
            monthly_payment = st.number_input("ชำระต่องวด (บาทต่อเดือน)", min_value=0, value=2000, step=500)
        submitted = st.form_submit_button("ประเมินความเสี่ยง")

    if submitted:
        # คำนวณอัตราส่วนชำระต่อรายได้ (DTI)
        dti_calculated = (monthly_payment / income) * 100 if income > 0 else 0
        
        # สร้าง DataFrame สำหรับป้อนข้อมูลเข้าโมเดล
        input_data = pd.DataFrame({
            'อาชีพ': [occupation],
            'วัตถุประสงค์': [purpose],
            'สถานภาพ': [status],
            'เพศ': [gender],
            'รายได้': [income],
            'วงเงินที่ขอ': [loan_amount],
            'อัตราส่วนชำระต่อรายได้': [dti_calculated]
        })
        
        # ทำนายความน่าจะเป็นของ NPL
        predict_proba = model.predict_proba(input_data)[0]
        npl_probability = predict_proba[1] # ความน่าจะเป็นที่จะเป็น NPL (คลาส 1)

        st.subheader("🎯 ผลการประเมิน")
        # แสดงผลตามระดับความเสี่ยง
        if npl_probability < 0.25:
            st.success(f"**เสี่ยงต่ำ** (ความน่าจะเป็น NPL: {npl_probability:.2%})", icon="✅")
        elif npl_probability < 0.50:
            st.info(f"**เสี่ยงปานกลาง** (ความน่าจะเป็น NPL: {npl_probability:.2%})", icon="🤔")
        elif npl_probability < 0.75:
            st.warning(f"**เสี่ยงสูง** (ความน่าจะเป็น NPL: {npl_probability:.2%})", icon="⚠️")
        else:
            st.error(f"**เสี่ยงสูงมาก** (ความน่าจะเป็น NPL: {npl_probability:.2%})", icon="🚨")
        
        st.progress(npl_probability) # แสดงแถบความคืบหน้า
        st.caption("แถบด้านบนแสดงความน่าจะเป็นที่ลูกค้ารายนี้จะกลายเป็นหนี้เสีย (NPL)")

# --- Chatbot ฝั่งขวา ---
with col_info:
    st.header("🤖 ผู้ช่วย AI ให้คำปรึกษา")
    st.write("พิมพ์คำถามด้านสินเชื่อหรือการประเมินความเสี่ยง แล้วรับคำตอบจาก AI ระดับ Gemini Flash") # อัปเดตข้อความ

    # กำหนดค่าเริ่มต้นของข้อความแชทใน session_state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            # System role สำหรับกำหนด Persona ของ AI แต่ไม่ถูกส่งตรงไปยัง API payload
            {"role": "system", "content": "คุณคือผู้ช่วยด้านสินเชื่อของสหกรณ์ ให้คำแนะนำเรื่อง NPL อย่างสุภาพและเข้าใจง่าย"},
            {"role": "assistant", "content": "สวัสดีครับ! มีอะไรให้ผมช่วยเกี่ยวกับสินเชื่อหรือการประเมิน NPL ไหมครับ?"}
        ]

    # แสดงข้อความแชทที่ผ่านมา (เริ่มจากข้อความของ assistant ตัวแรก)
    for msg in st.session_state.messages[1:]: 
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # รับ input จากผู้ใช้
    if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังตอบ..."):
                try:
                    # ดึง Gemini API Key จาก environment variables
                    gemini_api_key = os.getenv("GEMINI_API_KEY")
                    
                    if not gemini_api_key:
                        assistant_reply = "ไม่พบ GEMINI_API_KEY ในไฟล์ .env หรือ Environment Variables. กรุณาตั้งค่าก่อนใช้งาน."
                    else:
                        # เตรียมข้อความสำหรับการส่งไปยัง Gemini API
                        # แปลง 'assistant' role เป็น 'model' role สำหรับ Gemini
                        gemini_messages_for_api = []
                        for msg in st.session_state.messages:
                            if msg["role"] == "user":
                                gemini_messages_for_api.append({"role": "user", "parts": [{"text": msg["content"]}]})
                            elif msg["role"] == "assistant":
                                gemini_messages_for_api.append({"role": "model", "parts": [{"text": msg["content"]}]})
                            # ไม่ต้องส่ง 'system' role ไปใน payload โดยตรง

                        # กำหนด URL ของ Gemini API (สำหรับ gemini-2.0-flash)
                        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
                        
                        # สร้าง payload สำหรับ request
                        payload = {
                            "contents": gemini_messages_for_api,
                            "generationConfig": {
                                "temperature": 0.5, # ปรับ temperature ตามต้องการ
                            }
                        }

                        # ส่ง POST request ไปยัง Gemini API
                        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
                        response.raise_for_status() # ตรวจสอบข้อผิดพลาด HTTP status (เช่น 4xx, 5xx)
                        result = response.json()

                        # ดึงคำตอบจาก response ของ Gemini API
                        if result.get("candidates") and len(result["candidates"]) > 0 and \
                           result["candidates"][0].get("content") and \
                           len(result["candidates"][0]["content"].get("parts", [])) > 0:
                            assistant_reply = result["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            assistant_reply = "ไม่สามารถรับคำตอบจาก Gemini API ได้ (โครงสร้างการตอบกลับไม่ถูกต้อง หรือไม่มี candidates)."
                
                except requests.exceptions.RequestException as req_err:
                    assistant_reply = f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini API: {req_err}. ตรวจสอบ API Key และการเชื่อมต่ออินเทอร์เน็ตของคุณ."
                except json.JSONDecodeError:
                    assistant_reply = "เกิดข้อผิดพลาดในการถอดรหัส JSON จาก Gemini API (การตอบกลับไม่ถูกต้อง)."
                except Exception as e:
                    assistant_reply = f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}"

            # แสดงคำตอบของ AI และเพิ่มลงใน session_state
            st.markdown(assistant_reply)
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})





    #st.caption("แถบด้านบนแสดงความน่าจะเป็นที่ลูกค้ารายนี้จะกลายเป็นหนี้เสีย (NPL)")#.\venv\Scripts\activate#streamlit run app.py

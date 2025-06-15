import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import json
import numpy as np # Import numpy for handling potential division by zero

print("--- เริ่มกระบวนการฝึกโมเดล ---")

try:
    # Ensure your 'df.xlsx' has 'ชำระต่องวด' and 'รายได้' columns.
    df = pd.read_excel('df.xlsx')
    print("โหลดไฟล์ 'df.xlsx' สำเร็จ")

    # Convert target column 'สถานะNPL'
    df['สถานะNPL'] = df['สถานะNPL'].map({'N': 0, 'Y': 1})
    print("แปลงค่า 'สถานะNPL' เป็น 0/1 แล้ว")

    # --- EDITED: Create the Debt-to-Income Ratio column ---
    # This calculation must match the one in app.py
    # We replace infinity values (from division by zero) with 0.
    print("กำลังคำนวณ 'อัตราส่วนชำระต่อรายได้'...")
    df['อัตราส่วนชำระต่อรายได้'] = (df['ชำระต่องวด'] / df['รายได้']) * 100
    df.replace([np.inf, -np.inf], 0, inplace=True) # Handle cases where income might be 0
    df['อัตราส่วนชำระต่อรายได้'].fillna(0, inplace=True) # Handle potential NaN values
    print("สร้างคอลัมน์ 'อัตราส่วนชำระต่อรายได้' เรียบร้อยแล้ว")

except FileNotFoundError:
    print("ไม่พบไฟล์ 'df.xlsx'! กรุณาวางไฟล์ในโฟลเดอร์เดียวกับสคริปต์")
    exit()
except KeyError as e:
    print(f"เกิดข้อผิดพลาด: ไม่พบคอลัมน์ที่จำเป็นในไฟล์ Excel: {e}")
    print("กรุณาตรวจสอบว่าไฟล์ 'df.xlsx' มีคอลัมน์ 'ชำระต่องวด' และ 'รายได้'")
    exit()

# --- Prepare Data ---
TARGET_COLUMN = 'สถานะNPL'
# These feature names must exactly match the columns in your DataFrame
CATEGORICAL_FEATURES = ['อาชีพ', 'วัตถุประสงค์', 'สถานภาพ', 'เพศ']
NUMERICAL_FEATURES = ['รายได้', 'วงเงินที่ขอ', 'อัตราส่วนชำระต่อรายได้'] # This name is now consistent

# Check if all required columns exist
required_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN]
if not all(col in df.columns for col in required_cols):
    print(f"คำเตือน: มีบางคอลัมน์ขาดหายไปในไฟล์ Excel ของคุณ กรุณาตรวจสอบ: {required_cols}")
    exit()


X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
y = df[TARGET_COLUMN]

# --- Save options for UI ---
options = {col: X[col].unique().tolist() for col in CATEGORICAL_FEATURES}
with open('options.json', 'w', encoding='utf8') as f:
    json.dump(options, f, ensure_ascii=False, indent=4)
print("บันทึกตัวเลือกสำหรับ UI ลงใน 'options.json' เรียบร้อยแล้ว")

# --- Create Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ],
    remainder='passthrough'
)

# --- Create and Train Model Pipeline ---
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])
model.fit(X, y)
print("ฝึกโมเดลสำเร็จ!")

# --- Save the entire Pipeline ---
joblib.dump(model, 'pipeline.joblib')
print("บันทึก Pipeline ลงใน 'pipeline.joblib' เรียบร้อยแล้ว")
print("--- กระบวนการเสร็จสิ้น ---")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import json

print("--- เริ่มกระบวนการฝึกโมเดล ---")

# ... โค้ดส่วนบน ...

# --- 1. โหลดข้อมูลจาก Excel ---
try: # บรรทัด 'try:' จะไม่มีการเว้นวรรค หรือมีการเว้นวรรคเท่ากับบล็อกก่อนหน้า
    df = pd.read_excel('df.xlsx') # <--- บรรทัดนี้ **ต้อง** เว้นวรรคเข้ามา 4 ช่องว่าง (หรือ 1 tab) จาก 'try:'
    print("โหลดไฟล์ 'df.xlsx' สำเร็จ") # <--- บรรทัดนี้ **ต้อง** เว้นวรรคเข้ามาเท่ากับบรรทัดบน (4 ช่องว่างหรือ 1 tab)
except FileNotFoundError: # บรรทัด 'except:' จะต้องเว้นวรรคเท่ากับ 'try:'
    print("ไม่พบไฟล์ 'df.xlsx'! กรุณาวางไฟล์ในโฟลเดอร์เดียวกับสคริปต์") # <--- บรรทัดนี้ **ต้อง** เว้นวรรคเข้ามา 4 ช่องว่าง (หรือ 1 tab) จาก 'except:'
    exit()

# ... โค้ดส่วนที่เหลือ ...
# ... rest of your code
# --- 2. เตรียมข้อมูล ---
# **ปรับแก้ชื่อคอลัมน์เหล่านี้ให้ตรงกับไฟล์ Excel ของคุณ**
TARGET_COLUMN = 'NPL' # ชื่อคอลัมน์เป้าหมาย (0 หรือ 1)
CATEGORICAL_FEATURES = ['อาชีพ', 'วัตถุประสงค์', 'สถานภาพ', 'เพศ']
NUMERICAL_FEATURES = ['รายได้', 'วงเงินที่ขอ', 'อัตราส่วนชำระต่อรายได้']

# ตรวจสอบว่าคอลัมน์เป้าหมายมีค่าเป็น 0 และ 1
if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
    print(f"คำเตือน: คอลัมน์เป้าหมาย '{TARGET_COLUMN}' ควรมีแค่ค่า 0 และ 1")
    # ตัวอย่างการแปลง 'NPL'/'Non-NPL' เป็น 1/0
    # df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x == 'NPL' else 0)

X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
y = df[TARGET_COLUMN]

# --- 3. บันทึกตัวเลือกสำหรับ UI ---
# ดึงค่าที่ไม่ซ้ำกันจากแต่ละคอลัมน์เพื่อใช้สร้าง dropdown ในแอป
options = {col: X[col].unique().tolist() for col in CATEGORICAL_FEATURES}
with open('options.json', 'w', encoding='utf8') as f:
    json.dump(options, f, ensure_ascii=False, indent=4)
print("บันทึกตัวเลือกสำหรับ UI ลงใน 'options.json' เรียบร้อยแล้ว")

# --- 4. สร้าง Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ],
    remainder='passthrough' # เก็บ Column ที่เหลือไว้ (ถ้ามี)
)

# --- 5. สร้างและฝึกโมเดล Pipeline ---
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])
model.fit(X, y)
print("ฝึกโมเดลสำเร็จ!")

# --- 6. บันทึก Pipeline ทั้งหมด ---
joblib.dump(model, 'pipeline.joblib')
print("บันทึก Pipeline ลงใน 'pipeline.joblib' เรียบร้อยแล้ว")
print("--- กระบวนการเสร็จสิ้น ---")
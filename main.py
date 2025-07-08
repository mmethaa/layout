import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Smart Layout AI", page_icon="🏗️", layout="centered")

# -------------------- STYLING --------------------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f0f4f8;
    }
    .stButton>button {
        background: linear-gradient(to right, #0f4c75, #3282b8);
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 0.6em 2em;
    }
    .stMetricLabel, .stMetricValue {
        color: #1f2937 !important;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1em;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div style="background: linear-gradient(to right, #0f4c75, #3282b8); padding: 2rem; border-radius: 12px; text-align: center;">
    <h1 style="color: white;">Smart Layout Predictor 🏢</h1>
    <p style="color: #d9ecf2; font-size: 18px;">ระบบพยากรณ์ผังโครงการอสังหาฯ สำหรับ Developer และ Builder</p>
</div>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
df = pd.read_excel("updatedata.xlsx")
df.columns = df.columns.str.strip()
df['หลังต่อซอย'] = df['จำนวนหลัง'] / df['จำนวนซอย'].replace(0, 1)
df['%บ้านเดี่ยว'] = df['บ้านเดี่ยว'] / df['จำนวนหลัง'].replace(0, 1)
df['%บ้านแฝด'] = df['บ้านแฝด'] / df['จำนวนหลัง'].replace(0, 1)
df['%ทาวโฮม'] = df['ทาวโฮม'] / df['จำนวนหลัง'].replace(0, 1)
df['%พื้นที่ขาย'] = df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']
df['%พื้นที่สาธา'] = df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']

X_raw = df[['จังหวัด', 'เกรดโครงการ', 'พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน']]
y_ratio = pd.DataFrame({
    'สัดส่วนพื้นที่สาธา': df['%พื้นที่สาธา'],
    'สัดส่วนพื้นที่จัดจำหน่าย': df['%พื้นที่ขาย'],
    'สัดส่วนพื้นที่สวน': df['พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)'] / df['พื้นที่โครงการ(ตรม)'],
    'จำนวนหลังต่อไร่': df['จำนวนหลัง'] / (df['พื้นที่โครงการ(ตรม)'] / 1600),
    'สัดส่วนทาวโฮม': df['%ทาวโฮม'],
    'สัดส่วนบ้านแฝด': df['%บ้านแฝด'],
    'สัดส่วนบ้านเดี่ยว': df['%บ้านเดี่ยว'],
    'สัดส่วนอาคารพาณิชย์': df['อาคารพาณิชย์'] / df['จำนวนหลัง'].replace(0, 1)
})

X = pd.get_dummies(X_raw, columns=['จังหวัด', 'เกรดโครงการ', 'รูปร่างที่ดิน'])
X_train, _, y_train, _ = train_test_split(X, y_ratio, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)).fit(X_train, y_train)
avg_ซอยต่อหลัง = df.groupby('เกรดโครงการ')['หลังต่อซอย'].mean().to_dict()

# -------------------- FORM --------------------
st.markdown("## 📋 กรอกข้อมูลโครงการ")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        จังหวัด = st.selectbox("📍 จังหวัด", sorted(df['จังหวัด'].dropna().unique()))
        รูปร่าง = st.selectbox("🧱️ รูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].dropna().unique()))
    with col2:
        เกรด = st.selectbox("🏷️ เกรดโครงการ", sorted(df['เกรดโครงการ'].dropna().unique()))
        พื้นที่ = st.number_input("📀 พื้นที่โครงการ (ตร.ม.)", min_value=1000, value=30000, step=500)
    submitted = st.form_submit_button("🚀 เริ่มพยากรณ์")

# -------------------- PREDICT --------------------
if submitted:
    area = พื้นที่
    rai = area / 1600
    input_df = pd.DataFrame([{ 'จังหวัด': จังหวัด, 'เกรดโครงการ': เกรด, 'พื้นที่โครงการ(ตรม)': area, 'รูปร่างที่ดิน': รูปร่าง }])
    encoded = pd.get_dummies(input_df)
    for col in X.columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[X.columns]
    pred = model.predict(encoded)[0]

    พท_สาธา = pred[0] * area
    พท_ขาย = pred[1] * area
    พท_สวน = pred[2] * area
    หลังรวม = pred[3] * rai
    total = sum(pred[4:8]) or 1
    ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [หลังรวม * (r / total) for r in pred[4:8]]

    คำเตือน = None
    grade_high = ['MONTARA', 'PRIMAVILLA', 'RIVIERA', 'GRANDVILLE', 'PRIME', 'TASCANY', 'GRANDESSENCE', 'ESSENCE']
    grade_mid = ['PALMSPRING', 'VILLE', 'PARKVILLE', 'GARDENVILLE', 'PALMVILLE']
    grade_low = ['BELLA', 'PRIMO', 'WATTANALAI', 'PRIDE']

    if เกรด in grade_high:
        ทาวโฮม = บ้านแฝด = อาคารพาณิชย์ = 0
        บ้านเดี่ยว = หลังรวม
        คำเตือน = "🏡 เฉพาะบ้านเดี่ยว"
    elif เกรด in grade_mid:
        คำเตือน = "🏠 แบบผสมผสานตามแนวโน้ม"
    elif เกรด in grade_low:
        if บ้านเดี่ยว > หลังรวม * 0.2:
            บ้านเดี่ยว = หลังรวม * 0.2
        if อาคารพาณิชย์ > หลังรวม * 0.1:
            อาคารพาณิชย์ = 0
        ที่เหลือ = หลังรวม - บ้านเดี่ยว
        ทาวโฮม = ที่เหลือ * 0.7
        บ้านแฝด = ที่เหลือ * 0.3
        คำเตือน = "🏪 เน้นทาวน์โฮม/บ้านแฝด"

    ซอย = หลังรวม / avg_ซอยต่อหลัง.get(เกรด, 12)

    st.markdown("---")
    st.markdown("## 🌟 ผลลัพธ์พยากรณ์")
    if คำเตือน:
        st.warning(คำเตือน)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("พื้นที่สาธารณะ", f"{พท_สาธา:,.0f} ตร.ม.")
        st.metric("พื้นที่จัดจำหน่าย", f"{พท_ขาย:,.0f} ตร.ม.")
        st.metric("พื้นที่สวน", f"{พท_สวน:,.0f} ตร.ม.")
    with col2:
        st.metric("จำนวนแปลงรวม", f"{หลังรวม:,.0f} หลัง")
        st.metric("จำนวนซอย", f"{ซอย:,.0f} ซอย")

    st.markdown("### 🏡 แยกตามประเภศบ้าน")
    st.markdown(f"""
    - ทาวน์โฮม: **{ทาวโฮม:,.0f}** หลัง
    - บ้านแฝด: **{บ้านแฝด:,.0f}** หลัง
    - บ้านเดี่ยว: **{บ้านเดี่ยว:,.0f}** หลัง
    - อาคารพาณิชย์: **{อาคารพาณิชย์:,.0f}** หลัง
    """)

st.markdown("---")
st.caption("Developed by mmethaa | Smart Layout AI 🚀")


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Smart Layout AI", page_icon="🌇️", layout="centered")

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
    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 12px;
        padding: 1em;
        margin: 10px 0;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }
    div[data-testid="metric-container"] > label, div[data-testid="metric-container"] > div {
        color: #1f2937 !important;
        font-weight: 600;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Excel and prepare
sheet_name = "Sheet1"
df = pd.read_excel("layoutdata.xlsx", sheet_name=sheet_name)
df.columns = df.columns.str.strip()

# Convert sqm to sq.wah (1 ตร.ม. = 0.25 ตร.วา)
sqm_to_sqwah = 0.25
for col in ['พื้นที่โครงการ(ตรม)', 'พื้นที่จัดจำหน่าย(ตรม)', 'พื้นที่สาธา(ตรม)', 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)', 'พื้นที่ถนนรวม']:
    df[col] *= sqm_to_sqwah

# Define average unit area per house type (ตร.วา)
house_dims = {
    'ทาวโฮม': 5 * 16 * sqm_to_sqwah,
    'บ้านแฝด': 10 * 16 * sqm_to_sqwah,
    'บ้านเดี่ยว': 15 * 18 * sqm_to_sqwah
}

# Add total units column
df['total_units'] = df[['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']].sum(axis=1)

# Calculate proportions per house type
types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']
for t in types:
    df[f'{t}_prop'] = df[t] / df['total_units'].replace(0, 1)

df.fillna(0, inplace=True)

# Compute averages
avg_public_area_ratio = (df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']).mean()
avg_dist_area_ratio = (df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']).mean()
avg_road_area_ratio = (df['พื้นที่ถนนรวม'] / df['พื้นที่โครงการ(ตรม)']).mean()
avg_units_per_dist_area = (df['total_units'] / df['พื้นที่จัดจำหน่าย(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
avg_alley_per_unit = (df['จำนวนซอย'] / df['total_units']).replace([np.inf, -np.inf], np.nan).mean()

# Group proportions by grade and shape
grade_land_shape_proportions = df.groupby(['เกรดโครงการ','รูปร่างที่ดิน'])[[f'{t}_prop' for t in types]].mean()

# Detect exclusive บ้านเดี่ยว grades
grade_rules = {}
for g in df['เกรดโครงการ'].unique():
    sub = df[df['เกรดโครงการ'] == g]
    if (sub[['ทาวโฮม','บ้านแฝด','อาคารพาณิชย์']].sum().sum() == 0):
        grade_rules[g] = {'บ้านเดี่ยว': 1.0}

def predict(project_area_wah, grade, shape):
    dist_area = project_area_wah * avg_dist_area_ratio
    public_area = project_area_wah * avg_public_area_ratio
    road_area = project_area_wah * avg_road_area_ratio
    garden_area = dist_area * 0.05

    prop_df = grade_land_shape_proportions.loc[(grade, shape)] if (grade, shape) in grade_land_shape_proportions.index else None

    prop = {t: 0 for t in types}
    if grade in grade_rules:
        prop.update(grade_rules[grade])
    elif prop_df is not None:
        for t in types:
            prop[t] = prop_df.get(f'{t}_prop', 0)
    else:
        for t in types:
            prop[t] = df[f'{t}_prop'].mean()

    total_units_est = dist_area * avg_units_per_dist_area
    known = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว']
    total_area = sum(prop[t] * total_units_est * house_dims.get(t, 0) for t in known)

    scale = dist_area / total_area if total_area > 0 else 1
    result = {}
    for t in known:
        result[t] = round(prop[t] * total_units_est * scale)
    for t in ['บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']:
        result[t] = round(prop[t] * total_units_est)
    result['รวม'] = sum(result.values())
    result['ซอย'] = max(1, round(result['รวม'] * avg_alley_per_unit))

    return {
        'พื้นที่โครงการ (ตร.วา)': round(project_area_wah, 2),
        'พื้นที่สาธารณะ': round(public_area, 2),
        'พื้นที่จัดจำหน่าย': round(dist_area, 2),
        'พื้นที่สวน': round(garden_area, 2),
        'พื้นที่ถนน': round(road_area, 2),
        **{f'จำนวนแปลง ({k})': v for k, v in result.items() if k != 'รวม' and k != 'ซอย'},
        'จำนวนแปลง (รวม)': result['รวม'],
        'จำนวนซอย': result['ซอย']
    }

# UI
st.title("📐 Smart Layout AI")
st.subheader("ระบบพยากรณ์ผังจัดสรร")

with st.form("input"):
    area_input = st.number_input("กรอกพื้นที่โครงการ (หน่วย: ตร.วา)", min_value=1.0, value=10000.0)
    grade_input = st.selectbox("เลือกเกรดโครงการ", sorted(df['เกรดโครงการ'].unique()))
    shape_input = st.selectbox("เลือกรูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].unique()))
    submitted = st.form_submit_button("พยากรณ์")

if submitted:
    out = predict(area_input, grade_input, shape_input)
    st.subheader("📊 ผลลัพธ์การพยากรณ์")
    for k, v in out.items():
        st.write(f"{k}: {v}")

    st.divider()
    # Accuracy placeholder
    actual = df['total_units']
    pred = df['พื้นที่จัดจำหน่าย(ตรม)'] * avg_units_per_dist_area
    mep = np.mean(np.abs(actual - pred) / np.where(actual != 0, actual, 1)) * 100
    r2 = r2_score(actual, pred)
    st.metric("MEP (Mean Error %)", f"{mep:.2f}%")
    st.metric("R² Score", f"{r2:.4f}")

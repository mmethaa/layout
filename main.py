import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Load Data ---
try:
    df = pd.read_excel('layoutdata.xlsx', sheet_name='Sheet1')
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    st.error("ไม่พบไฟล์ layoutdata.xlsx หรือ Sheet1")
    st.stop()

# --- 2. Unit Conversion (sqm ➝ sq. wah) ---
sqm_to_sqwah = 0.25
for col in ['พื้นที่โครงการ(ตรม)', 'พื้นที่จัดจำหน่าย(ตรม)', 'พื้นที่สาธา(ตรม)', 'พื้นที่ถนนรวม', 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)']:
    if col in df.columns:
        df[col] *= sqm_to_sqwah

# --- 3. Setup House Area ---
AREA_TH = 5 * 16 * sqm_to_sqwah
AREA_BA = 10 * 16 * sqm_to_sqwah
AREA_BD = 15 * 18 * sqm_to_sqwah

house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']

# --- 4. Aggregate Ratios ---
avg_public_area_ratio = df['พื้นที่สาธา(ตรม)'].sum() / df['พื้นที่โครงการ(ตรม)'].sum()
avg_distributable_area_ratio = df['พื้นที่จัดจำหน่าย(ตรม)'].sum() / df['พื้นที่โครงการ(ตรม)'].sum()
avg_road_area_ratio = df['พื้นที่ถนนรวม'].sum() / df['พื้นที่โครงการ(ตรม)'].sum()
avg_units_per_dist_area = df['จำนวนหลัง'].sum() / df['พื้นที่จัดจำหน่าย(ตรม)'].sum()
avg_alley_per_unit = df['จำนวนซอย'].sum() / df['จำนวนหลัง'].sum()
avg_alley_per_dist_area = df['จำนวนซอย'].sum() / df['พื้นที่จัดจำหน่าย(ตรม)'].sum()

# --- 5. Grade Rules ---
grade_rules = {g: {'บ้านเดี่ยว': 1.0} for g in df['เกรดโครงการ'].unique() if ((df[df['เกรดโครงการ'] == g][['ทาวโฮม','บ้านแฝด','อาคารพาณิชย์']].sum().sum()) == 0)}

# --- 6. Proportion Table ---
for h in house_types:
    if h in df.columns:
        df[f'{h}_prop'] = df[h] / df['จำนวนหลัง'].replace(0, np.nan)
df.fillna(0, inplace=True)
grade_land_shape_proportions = df.groupby(['เกรดโครงการ', 'รูปร่างที่ดิน'])[[f'{h}_prop' for h in house_types]].mean()

# --- 7. Prediction Function ---
def predict(project_area, land_shape, grade, province):
    public_area = project_area * avg_public_area_ratio
    dist_area = project_area * avg_distributable_area_ratio
    garden_area = dist_area * 0.05
    road_area = project_area * avg_road_area_ratio

    estimated_units = int(dist_area * avg_units_per_dist_area)

    props = grade_land_shape_proportions.loc[(grade, land_shape)] if (grade, land_shape) in grade_land_shape_proportions.index else pd.Series()
    if props.empty:
        props = pd.Series({f'{h}_prop': df[f'{h}_prop'].mean() for h in house_types if f'{h}_prop' in df.columns})
    props = props / props.sum()

    units = {h: round(estimated_units * props.get(f'{h}_prop', 0)) for h in house_types}

    area_used = units['ทาวโฮม'] * AREA_TH + units['บ้านแฝด'] * AREA_BA + units['บ้านเดี่ยว'] * AREA_BD
    if area_used > 0:
        scale = dist_area / area_used
        for h, a in zip(['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว'], [AREA_TH, AREA_BA, AREA_BD]):
            units[h] = max(0, round(units[h] * scale))

    total_units = sum(units.values())
    alleys = max(1, round(total_units * avg_alley_per_unit))

    return {
        'พื้นที่โครงการ (ตร.วา)': round(project_area, 2),
        'พื้นที่สาธารณะ (ตร.วา)': round(public_area, 2),
        'พื้นที่ขาย (ตร.วา)': round(dist_area, 2),
        'พื้นที่สวน (ตร.วา)': round(garden_area, 2),
        'พื้นที่ถนน (ตร.วา)': round(road_area, 2),
        'จำนวนแปลง (ทาวน์โฮม)': units['ทาวโฮม'],
        'จำนวนแปลง (บ้านแฝด)': units['บ้านแฝด'],
        'จำนวนแปลง (บ้านเดี่ยว)': units['บ้านเดี่ยว'],
        'จำนวนแปลง (บ้านเดี่ยว3ชั้น)': units['บ้านเดี่ยว3ชั้น'],
        'จำนวนแปลง (อาคารพาณิชย์)': units['อาคารพาณิชย์'],
        'จำนวนแปลงรวม': total_units,
        'จำนวนซอย': alleys
    }

# --- 8. Metrics ---
def calculate_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mep = np.mean(np.abs(actual - predicted) / np.maximum(actual, 1)) * 100
    r2 = r2_score(actual, predicted)
    return mep, r2

# --- 9. UI ---
st.title("📐 Smart Layout Predictor")

project_area = st.number_input("พื้นที่โครงการ (ตร.วา)", 1000.0, 100000.0, 40000.0, step=500.0)
land_shape = st.selectbox("รูปร่างที่ดิน", df['รูปร่างที่ดิน'].unique())
grade = st.selectbox("เกรดโครงการ", df['เกรดโครงการ'].unique())
province = st.selectbox("จังหวัด", df['จังหวัด'].unique())

if st.button("ทำนายผังโครงการ"):
    result = predict(project_area, land_shape, grade, province)
    st.subheader("🔍 ผลการทำนาย")
    st.dataframe(pd.DataFrame(result.items(), columns=['รายการ', 'ค่าทำนาย']), use_container_width=True)

    actual = df['จำนวนหลัง'][:10]
    pred = df['พื้นที่จัดจำหน่าย(ตรม)'][:10] * avg_units_per_dist_area
    mep, r2 = calculate_metrics(actual, pred)
    st.metric("MEP", f"{mep:.2f}%")
    st.metric("R²", f"{r2:.2f}")

import streamlit as st
import pandas as pd
import numpy as np
# Import sklearn modules if you plan to use them for model training later,
# currently the prediction logic is based on averages/ratios.
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Smart Layout AI",
    page_icon="🌇️",
    layout="centered",
    initial_sidebar_state="expanded" # Keep this here, as it's the first call
)

st.markdown("""
    <style>
    html, body, [class*="css"] {
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

# --- 1. Load Data ---
@st.cache_data # Cache the data loading to improve performance
def load_data():
    try:
        # !!! สำคัญมาก: เปลี่ยนกลับมาอ่านไฟล์ CSV !!!
        df = pd.read_csv('layoutdata.xlsx - Sheet1.csv')
        df.columns = df.columns.str.strip() # Strip whitespace from column names

        # Calculate Total Units if not already present
        # ตรวจสอบว่าคอลัมน์มีอยู่จริงก่อนคำนวณ
        required_house_cols = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']
        if all(col in df.columns for col in required_house_cols):
            if 'จำนวนหลัง' not in df.columns:
                df['จำนวนหลัง'] = df['ทาวโฮม'] + df['บ้านแฝด'] + df['บ้านเดี่ยว'] + df['บ้านเดี่ยว3ชั้น'] + df['อาคารพาณิชย์']
        else:
            st.error("ข้อผิดพลาด: คอลัมน์ประเภทบ้าน (ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, บ้านเดี่ยว3ชั้น, อาคารพาณิชย์) ไม่ครบถ้วนในไฟล์ CSV")
            st.stop()

        # House types list for iteration
        house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']

        # Calculate proportion for each house type
        df['total_houses_for_prop'] = df[required_house_cols].sum(axis=1)
        for h_type in house_types:
            # Handle division by zero for proportions
            df[f'{h_type}_prop'] = df.apply(lambda row: row[h_type] / row['total_houses_for_prop'] if row['total_houses_for_prop'] > 0 else 0, axis=1)
        df.fillna(0, inplace=True) # Fill NaN proportions with 0

        return df
    except FileNotFoundError:
        st.error("ข้อผิดพลาด: ไม่พบไฟล์ 'layoutdata.xlsx - Sheet1.csv' กรุณาตรวจสอบให้แน่ใจว่าไฟล์อยู่ในโฟลเดอร์เดียวกันกับสคริปต์")
        st.stop()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดขณะโหลดหรือประมวลผลข้อมูล: {e}")
        st.stop()

df = load_data()

# --- 2. Pre-calculate average ratios and proportions from historical data ---
# !!! สำคัญ: ไม่ต้องแปลง df เป็นตรว. ที่นี่ เพราะ predict_project_layout_internal ต้องการตรม. !!!
# Ratios for area calculations (ใช้ค่าจาก ตรม. ใน df โดยตรง)
if 'พื้นที่สาธา(ตรม)' in df.columns and 'พื้นที่โครงการ(ตรม)' in df.columns:
    avg_public_area_ratio = (df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_public_area_ratio = 0.333
    st.warning("ไม่พบหรือไม่สมบูรณ์คอลัมน์ 'พื้นที่สาธา(ตรม)' หรือ 'พื้นที่โครงการ(ตรม)' ใช้ค่าเฉลี่ยเริ่มต้นสำหรับพื้นที่สาธารณะ")

if 'พื้นที่จัดจำหน่าย(ตรม)' in df.columns and 'พื้นที่โครงการ(ตรม)' in df.columns:
    avg_distributable_area_ratio = (df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_distributable_area_ratio = 0.667
    st.warning("ไม่พบหรือไม่สมบูรณ์คอลัมน์ 'พื้นที่จัดจำหน่าย(ตรม)' หรือ 'พื้นที่โครงการ(ตรม)' ใช้ค่าเฉลี่ยเริ่มต้นสำหรับพื้นที่จัดจำหน่าย")

if 'พื้นที่ถนนรวม' in df.columns and 'พื้นที่โครงการ(ตรม)' in df.columns:
    avg_road_area_ratio = (df['พื้นที่ถนนรวม'] / df['พื้นที่โครงการ(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_road_area_ratio = 0.30
    st.warning("ไม่พบหรือไม่สมบูรณ์คอลัมน์ 'พื้นที่ถนนรวม' หรือ 'พื้นที่โครงการ(ตรม)' ใช้ค่าเฉลี่ยเริ่มต้นสำหรับพื้นที่ถนน")

# Average area per unit for each type (these are in SQM, as per original intention)
AREA_TH = 5 * 16  # ทาวน์โฮม (80 sqm)
AREA_BA = 10 * 16 # บ้านแฝด (160 sqm) - ใช้ค่าตามที่คุณให้มาใน snippet ล่าสุด
AREA_BD = 15 * 18 # บ้านเดี่ยว (270 sqm)

# Average units per distributable area (overall)
if 'จำนวนหลัง' in df.columns and 'พื้นที่จัดจำหน่าย(ตรม)' in df.columns:
    avg_units_per_dist_area = (df['จำนวนหลัง'] / df['พื้นที่จัดจำหน่าย(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_units_per_dist_area = 0.005
    st.warning("ไม่พบหรือไม่สมบูรณ์คอลัมน์ 'จำนวนหลัง' หรือ 'พื้นที่จัดจำหน่าย(ตรม)' ใช้ค่าเฉลี่ยเริ่มต้นสำหรับจำนวนหลังต่อพื้นที่จัดจำหน่าย")

# House types list for iteration (ensure correct spelling)
house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']

# Group by 'เกรดโครงการ' and 'รูปร่างที่ดิน' to get average proportions
if 'เกรดโครงการ' in df.columns and 'รูปร่างที่ดิน' in df.columns:
    grade_land_shape_proportions = df.groupby(['เกรดโครงการ', 'รูปร่างที่ดิน'])[
        [f'{h_type}_prop' for h_type in house_types]
    ].mean()
else:
    st.error("ไม่พบหรือไม่สมบูรณ์คอลัมน์ 'เกรดโครงการ' หรือ 'รูปร่างที่ดิน' ในไฟล์ CSV ไม่สามารถคำนวณสัดส่วนตามเกรด/รูปร่างที่ดินได้")
    st.stop()


# Rules for specific grades (based on initial observation and user request)
# Reverted to explicit rules as discussed to ensure LUXURY only has บ้านเดี่ยว.
grade_rules = {
    'LUXURY': {'ทาวโฮม': 0, 'บ้านแฝด': 0, 'บ้านเดี่ยว3ชั้น': 0, 'อาคารพาณิชย์': 0},
    'PREMIUM': {'ทาวโฮม': 0, 'บ้านเดี่ยว3ชั้น': 0, 'อาคารพาณิชย์': 0}
}

# Average number of alleys per total units
if 'จำนวนซอย' in df.columns and 'จำนวนหลัง' in df.columns:
    avg_alley_per_unit = (df['จำนวนซอย'] / df['จำนวนหลัง']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_alley_per_unit = 0.05
    st.warning("ไม่พบหรือไม่สมบูรณ์คอลัมน์ 'จำนวนซอย' หรือ 'จำนวนหลัง' ใช้ค่าเฉลี่ยเริ่มต้นสำหรับจำนวนซอยต่อจำนวนหลัง")

if 'จำนวนซอย' in df.columns and 'พื้นที่จัดจำหน่าย(ตรม)' in df.columns:
    avg_alley_per_dist_area = (df['จำนวนซอย'] / df['พื้นที่จัดจำหน่าย(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_alley_per_dist_area = 0.0001
    st.warning("ไม่พบหรือไม่สมบูรณ์คอลัมน์ 'จำนวนซอย' หรือ 'พื้นที่จัดจำหน่าย(ตรม)' ใช้ค่าเฉลี่ยเริ่มต้นสำหรับจำนวนซอยต่อพื้นที่จัดจำหน่าย")

st.success("✅ ข้อมูลพร้อมแล้วสำหรับการทำนายผังโครงการ")

# --- Helper function for metrics ---
def calculate_metrics(actual_values, predicted_values):
    """Calculates MEP and R-squared, handling potential NaNs and zero actuals for MEP."""
    actual_values = np.array(actual_values).flatten()
    predicted_values = np.array(predicted_values).flatten()

    valid_indices = ~np.isnan(actual_values) & ~np.isnan(predicted_values)
    actual_values = actual_values[valid_indices]
    predicted_values = predicted_values[valid_indices]

    if len(actual_values) == 0:
        return {'MEP': np.nan, 'R2': np.nan}

    if np.all(actual_values == 0) and np.all(predicted_values == 0):
        return {'MEP': 0.0, 'R2': 1.0}
    
    diff_abs_percent = []
    for i in range(len(actual_values)):
        if actual_values[i] != 0:
            diff_abs_percent.append(np.abs((actual_values[i] - predicted_values[i]) / actual_values[i]))
        elif predicted_values[i] != 0:
            diff_abs_percent.append(1.0) 

    mep = np.mean(diff_abs_percent) * 100 if diff_abs_percent else 0.0

    if np.var(actual_values) == 0:
        r2 = 1.0 if np.all(actual_values == predicted_values) else 0.0
    else:
        r2 = r2_score(actual_values, predicted_values)

    return {'MEP': mep, 'R2': r2}

# --- 3. Prediction Function (Internal, works with SQM) ---
def predict_project_layout_internal(
    project_area_sqm_input: float,
    land_shape: str,
    project_grade: str,
    province: str
) -> dict:
    project_area_sqm = project_area_sqm_input

    predicted_public_area = project_area_sqm * avg_public_area_ratio
    predicted_distributable_area = project_area_sqm * avg_distributable_area_ratio
    predicted_garden_area = predicted_distributable_area * 0.05
    predicted_road_area = project_area_sqm * avg_road_area_ratio

    predicted_units = {h: 0 for h in house_types}
    total_predicted_units = 0

    proportions_key = (project_grade, land_shape)
    proportions_df_row = None
    if proportions_key in grade_land_shape_proportions.index:
        proportions_df_row = grade_land_shape_proportions.loc[proportions_key].to_frame().T

    if project_grade in grade_rules:
        # Apply grade-specific exclusions first
        for h_type_to_exclude, _ in grade_rules[project_grade].items():
            predicted_units[h_type_to_exclude] = 0
        
        remaining_house_types = [h for h in house_types if predicted_units[h] == 0 or h not in grade_rules[project_grade]]
        
        # Ensure 'บ้านเดี่ยว' is the only remaining type for LUXURY if that's the rule
        if project_grade == 'LUXURY':
            remaining_house_types = ['บ้านเดี่ยว']
            
        estimated_total_units_from_area = 0
        if not np.isnan(avg_units_per_dist_area) and avg_units_per_dist_area > 0:
            estimated_total_units_from_area = round(predicted_distributable_area * avg_units_per_dist_area)

        current_proportions = np.array([])
        if proportions_df_row is not None and not proportions_df_row.empty:
            # Use specific proportions if available for the remaining types
            cols_for_prop = [f'{h_type}_prop' for h_type in remaining_house_types if f'{h_type}_prop' in proportions_df_row.columns]
            if cols_for_prop:
                current_proportions = proportions_df_row[cols_for_prop].values.flatten()
                if current_proportions.sum() > 0:
                    current_proportions /= current_proportions.sum()
        
        if current_proportions.sum() == 0 and len(remaining_house_types) > 0:
            # Fallback to general average for remaining types if specific or normalized sum is zero
            general_avg_props_remaining = df[[f'{h_type}_prop' for h_type in remaining_house_types]].mean()
            if general_avg_props_remaining.sum() > 0:
                current_proportions = general_avg_props_remaining.values / general_avg_props_remaining.sum()
            elif len(remaining_house_types) > 0: # Last resort: equal distribution
                 current_proportions = np.ones(len(remaining_house_types)) / len(remaining_house_types)

        temp_predicted_remaining_units = {}
        for i, h_type in enumerate(remaining_house_types):
            if i < len(current_proportions):
                temp_predicted_remaining_units[h_type] = round(estimated_total_units_from_area * current_proportions[i])
        
        for h_type in remaining_house_types:
            predicted_units[h_type] = temp_predicted_remaining_units.get(h_type, 0)

    else: # No specific grade rules, use average proportions for the given grade/land shape
        estimated_total_units_from_area = 0
        if not np.isnan(avg_units_per_dist_area) and avg_units_per_dist_area > 0:
            estimated_total_units_from_area = round(predicted_distributable_area * avg_units_per_dist_area)
            
        if proportions_df_row is not None and not proportions_df_row.empty:
            for h_type in house_types:
                if f'{h_type}_prop' in proportions_df_row.columns:
                    predicted_units[h_type] = round(estimated_total_units_from_area * proportions_df_row[f'{h_type}_prop'].iloc[0])
        else:
            general_avg_props = df[[f'{h_type}_prop' for h_type in house_types]].mean()
            if general_avg_props.sum() > 0:
                general_avg_props /= general_avg_props.sum()
            for h_type in house_types:
                predicted_units[h_type] = round(estimated_total_units_from_area * general_avg_props[f'{h_type}_prop'])
    
    total_predicted_units = sum(predicted_units.values())

    # Refine units based on area consumed by primary types (TH, BA, BD)
    # This step should be applied after all type distribution logic.
    total_area_consumed_by_known_types = (
        predicted_units['ทาวโฮม'] * AREA_TH +
        predicted_units['บ้านแฝด'] * AREA_BA +
        predicted_units['บ้านเดี่ยว'] * AREA_BD
    )
    
    if total_area_consumed_by_known_types > 0 and predicted_distributable_area > 0:
        scale_factor = predicted_distributable_area / total_area_consumed_by_known_types
        for h_type in ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว']:
            predicted_units[h_type] = round(predicted_units[h_type] * scale_factor)
    
    total_predicted_units = sum(predicted_units.values()) # Recalculate total after scaling

    predicted_alleys = 0
    if total_predicted_units > 0 and not np.isnan(avg_alley_per_unit):
        predicted_alleys = round(total_predicted_units * avg_alley_per_unit)
    elif predicted_distributable_area > 0 and not np.isnan(avg_alley_per_dist_area):
        predicted_alleys = round(predicted_distributable_area * avg_alley_per_dist_area)
    
    predicted_alleys = max(1, predicted_alleys)

    return {
        'พื้นที่โครงการ (ตรม.)': round(project_area_sqm, 2),
        'พื้นที่สาธา (ตรม.)': round(predicted_public_area, 2),
        'พื้นที่จัดจำหน่าย (ตรม.)': round(predicted_distributable_area, 2),
        'พื้นที่สวน (ตรม.)': round(predicted_garden_area, 2),
        'พื้นที่ถนนรวม (ตรม.)': round(predicted_road_area, 2),
        'จำนวนแปลง (ทาวน์โฮม)': max(0, predicted_units['ทาวโฮม']),
        'จำนวนแปลง (บ้านแฝด)': max(0, predicted_units['บ้านแฝด']),
        'จำนวนแปลง (บ้านเดี่ยว)': max(0, predicted_units['บ้านเดี่ยว']),
        'จำนวนแปลง (บ้านเดี่ยว3ชั้น)': max(0, predicted_units['บ้านเดี่ยว3ชั้น']),
        'จำนวนแปลง (อาคารพาณิชย์)': max(0, predicted_units['อาคารพาณิชย์']),
        'จำนวนแปลง (รวม)': max(0, total_predicted_units),
        'จำนวนซอย': max(1, predicted_alleys)
    }

# --- Calculate Model Performance Metrics on Historical Data ---
actual_total_units = []
predicted_total_units_for_metrics = []
actual_distributable_areas = []
predicted_distributable_areas_for_metrics = []
actual_alleys = []
predicted_alleys_for_metrics = []

for index, row in df.iterrows():
    # Ensure necessary columns are present in the row before calling predict_project_layout_internal
    required_cols_for_prediction = [
        'พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด',
        'จำนวนหลัง', 'พื้นที่จัดจำหน่าย(ตรม)', 'จำนวนซอย'
    ]
    
    if all(col in row.index for col in required_cols_for_prediction): # Check if columns exist in the row's index
        pred_metrics = predict_project_layout_internal(
            project_area_sqm_input=row['พื้นที่โครงการ(ตรม)'],
            land_shape=row['รูปร่างที่ดิน'],
            project_grade=row['เกรดโครงการ'],
            province=row['จังหวัด']
        )
        
        actual_total_units.append(row['จำนวนหลัง'])
        predicted_total_units_for_metrics.append(pred_metrics['จำนวนแปลง (รวม)'])
        
        actual_distributable_areas.append(row['พื้นที่จัดจำหน่าย(ตรม)'])
        predicted_distributable_areas_for_metrics.append(pred_metrics['พื้นที่จัดจำหน่าย (ตรม.)'])
        
        actual_alleys.append(row['จำนวนซอย'])
        predicted_alleys_for_metrics.append(pred_metrics['จำนวนซอย'])
    else:
        st.warning(f"ข้ามแถวที่ {index} ในข้อมูลย้อนหลังเนื่องจากข้อมูลไม่ครบถ้วนสำหรับการคำนวณ Metrics หรือชื่อคอลัมน์ไม่ถูกต้อง")


metrics_total_units = calculate_metrics(actual_total_units, predicted_total_units_for_metrics)
metrics_dist_area = calculate_metrics(actual_distributable_areas, predicted_distributable_areas_for_metrics)
metrics_alleys = calculate_metrics(actual_alleys, predicted_alleys_for_metrics)


# --- 4. Main Prediction Function (User-facing, handles SQW) ---
def predict_project_layout_sqw(
    project_area_sqw_input: float, # Input is in Square Wah
    land_shape: str,
    project_grade: str,
    province: str
) -> dict:
    """
    Predicts layout metrics, converting input from Square Wah to Square Meters internally
    and converting results back to Square Wah for display.
    """
    # Convert input from Square Wah to Square Meters for internal calculations
    project_area_sqm = project_area_sqw_input * 4

    # Call the internal prediction function that uses SQM
    results_sqm = predict_project_layout_internal(
        project_area_sqm_input=project_area_sqm,
        land_shape=land_shape,
        project_grade=project_grade,
        province=province
    )

    # Convert relevant results back to Square Wah for display
    results_sqw = {
        'พื้นที่โครงการ (ตรว.)': round(project_area_sqw_input, 2),
        'พื้นที่สาธา (ตรว.)': round(results_sqm['พื้นที่สาธา (ตรม.)'] / 4, 2),
        'พื้นที่จัดจำหน่าย (ตรว.)': round(results_sqm['พื้นที่จัดจำหน่าย (ตรม.)'] / 4, 2),
        'พื้นที่สวน (ตรว.)': round(results_sqm['พื้นที่สวน (ตรม.)'] / 4, 2),
        'พื้นที่ถนนรวม (ตรว.)': round(results_sqm['พื้นที่ถนนรวม (ตรม.)'] / 4, 2),
        'จำนวนแปลง (ทาวน์โฮม)': results_sqm['จำนวนแปลง (ทาวน์โฮม)'],
        'จำนวนแปลง (บ้านแฝด)': results_sqm['จำนวนแปลง (บ้านแฝด)'],
        'จำนวนแปลง (บ้านเดี่ยว)': results_sqm['จำนวนแปลง (บ้านเดี่ยว)'],
        'จำนวนแปลง (บ้านเดี่ยว3ชั้น)': results_sqm['จำนวนแปลง (บ้านเดี่ยว3ชั้น)'],
        'จำนวนแปลง (อาคารพาณิชย์)': results_sqm['จำนวนแปลง (อาคารพาณิชย์)'],
        'จำนวนแปลง (รวม)': results_sqm['จำนวนแปลง (รวม)'],
        'จำนวนซอย': results_sqm['จำนวนซอย']
    }
    return results_sqw


# --- 5. Streamlit UI ---
st.title("🏡 การทำนายผังโครงการใหม่")
st.markdown("โปรดกรอกข้อมูลสำหรับโครงการใหม่เพื่อรับการทำนายผังและจำนวนบ้าน.")

# Display Model Performance Metrics in the sidebar
st.sidebar.header("📊 ประสิทธิภาพโมเดล (จากการทำนายข้อมูลย้อนหลัง)")
st.sidebar.metric(label="จำนวนแปลง (รวม) - ค่าคลาดเคลื่อนเฉลี่ย (MEP)", value=f"{metrics_total_units['MEP']:.2f}%" if not np.isnan(metrics_total_units['MEP']) else "N/A")
st.sidebar.metric(label="จำนวนแปลง (รวม) - R-squared ($R^2$)", value=f"{metrics_total_units['R2']:.2f}" if not np.isnan(metrics_total_units['R2']) else "N/A")
st.sidebar.markdown("---")
st.sidebar.metric(label="พื้นที่จัดจำหน่าย - ค่าคลาดเคลื่อนเฉลี่ย (MEP)", value=f"{metrics_dist_area['MEP']:.2f}%" if not np.isnan(metrics_dist_area['MEP']) else "N/A")
st.sidebar.metric(label="พื้นที่จัดจำหน่าย - R-squared ($R^2$)", value=f"{metrics_dist_area['R2']:.2f}" if not np.isnan(metrics_dist_area['R2']) else "N/A")
st.sidebar.markdown("---")
st.sidebar.metric(label="จำนวนซอย - ค่าคลาดเคลื่อนเฉลี่ย (MEP)", value=f"{metrics_alleys['MEP']:.2f}%" if not np.isnan(metrics_alleys['MEP']) else "N/A")
st.sidebar.metric(label="จำนวนซอย - R-squared ($R^2$)", value=f"{metrics_alleys['R2']:.2f}" if not np.isnan(metrics_alleys['R2']) else "N/A")


# Get unique values for dropdowns from loaded data
unique_land_shapes = df['รูปร่างที่ดิน'].unique().tolist() if 'รูปร่างที่ดิน' in df.columns else ["ไม่พบข้อมูล"]
unique_grades = df['เกรดโครงการ'].unique().tolist() if 'เกรดโครงการ' in df.columns else ["ไม่พบข้อมูล"]
unique_provinces = df['จังหวัด'].unique().tolist() if 'จังหวัด' in df.columns else ["ไม่พบข้อมูล"]


# Input widgets
st.header("ข้อมูลโครงการใหม่")
col1, col2 = st.columns(2)
with col1:
    project_area_sqw = st.number_input(
        "พื้นที่โครงการ (ตรว.)",
        min_value=250.0,
        max_value=250000.0,
        value=12500.0,
        step=250.0,
        help="ป้อนพื้นที่รวมของโครงการใหม่เป็นตารางวา"
    )
with col2:
    land_shape = st.selectbox(
        "รูปร่างที่ดิน",
        options=unique_land_shapes,
        help="เลือกรูปร่างที่ดินของโครงการใหม่"
    )
    if "ไม่พบข้อมูล" in unique_land_shapes:
        st.warning("ไม่พบข้อมูลรูปร่างที่ดินในไฟล์ CSV กรุณาตรวจสอบคอลัมน์ 'รูปร่างที่ดิน'")


col3, col4 = st.columns(2)
with col3:
    project_grade = st.selectbox(
        "เกรดโครงการ",
        options=unique_grades,
        help="เลือกเกรดของโครงการ (เช่น PREMIUM, LUXURY, BELLA)"
    )
    if "ไม่พบข้อมูล" in unique_grades:
        st.warning("ไม่พบข้อมูลเกรดโครงการในไฟล์ CSV กรุณาตรวจสอบคอลัมน์ 'เกรดโครงการ'")

with col4:
    province = st.selectbox(
        "จังหวัด",
        options=unique_provinces,
        help="เลือกจังหวัดที่โครงการตั้งอยู่ (ปัจจุบันไม่ได้ใช้ในการคำนวณโดยตรง)"
    )
    if "ไม่พบข้อมูล" in unique_provinces:
        st.warning("ไม่พบข้อมูลจังหวัดในไฟล์ CSV กรุณาตรวจสอบคอลัมน์ 'จังหวัด'")


if st.button("ทำนายผังโครงการ"):
    if "ไม่พบข้อมูล" in [land_shape, project_grade, province]:
        st.error("กรุณาแก้ไขข้อผิดพลาดเกี่ยวกับการไม่พบข้อมูลใน dropdowns ก่อนทำการทำนาย")
    elif project_area_sqw <= 0:
        st.error("กรุณาป้อน 'พื้นที่โครงการ' ที่มากกว่า 0.")
    else:
        with st.spinner("กำลังคำนวณ..."):
            predicted_results = predict_project_layout_sqw(
                project_area_sqw_input=project_area_sqw,
                land_shape=land_shape,
                project_grade=project_grade,
                province=province
            )
        
        st.success("ทำนายผลสำเร็จ!")
        st.header("ผลการทำนาย")
        
        results_df = pd.DataFrame(predicted_results.items(), columns=["ตัวชี้วัด", "ค่าที่ทำนาย"])
        st.dataframe(results_df, hide_index=True)

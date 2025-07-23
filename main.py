import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
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

sheet_name = "Sheet1"
df = pd.read_excel("layoutdata.xlsx", sheet_name=sheet_name)
df.columns = df.columns.str.strip()

# Convert all area units from sqm to sq.wah
sqm_to_sqwah = 0.25
df['พื้นที่โครงการ(ตรม)'] *= sqm_to_sqwah
df['พื้นที่จัดจำหน่าย(ตรม)'] *= sqm_to_sqwah
df['พื้นที่สาธา(ตรม)'] *= sqm_to_sqwah
df['พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)'] *= sqm_to_sqwah
df['พื้นที่ถนนรวม'] *= sqm_to_sqwah

# House dimension conversion
house_dims = {
    'ทาวโฮม': (5, 16),
    'บ้านแฝด': (10, 16),
    'บ้านเดี่ยว': (15, 18)
}

for htype, (w, l) in house_dims.items():
    df[f'ความกว้าง({htype})'] = w
    df[f'ความยาว({htype})'] = l
    df[f'พื้นที่เฉลี่ย({htype})'] = w * l

# Add area ratios
house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']
for h in house_types:
    col_name = f'{h}_prop'
    df[col_name] = df[h] / df['จำนวนหลัง'].replace(0, 1)

# Add dummy rules for certain grades that only allow บ้านเดี่ยว
grade_rules = {
    g: {'บ้านเดี่ยว': 1.0} for g in df['เกรดโครงการ'].unique() if ((df[df['เกรดโครงการ'] == g][['ทาวโฮม','บ้านแฝด','อาคารพาณิชย์']].sum().sum()) == 0)
}

avg_public_area_ratio = df['พื้นที่สาธา(ตรม)'].sum() / df['พื้นที่โครงการ(ตรม)'].sum()
avg_distributable_area_ratio = df['พื้นที่จัดจำหน่าย(ตรม)'].sum() / df['พื้นที่โครงการ(ตรม)'].sum()
avg_road_area_ratio = df['พื้นที่ถนนรวม'].sum() / df['พื้นที่โครงการ(ตรม)'].sum()
avg_units_per_dist_area = df['จำนวนหลัง'].sum() / df['พื้นที่จัดจำหน่าย(ตรม)'].sum()
avg_alley_per_unit = df['จำนวนซอย'].sum() / df['จำนวนหลัง'].sum()
avg_alley_per_dist_area = df['จำนวนซอย'].sum() / df['พื้นที่จัดจำหน่าย(ตรม)'].sum()

AREA_TH = house_dims['ทาวโฮม'][0] * house_dims['ทาวโฮม'][1]
AREA_BA = house_dims['บ้านแฝด'][0] * house_dims['บ้านแฝด'][1]
AREA_BD = house_dims['บ้านเดี่ยว'][0] * house_dims['บ้านเดี่ยว'][1]

# Create mapping for grade + shape proportions
grade_land_shape_proportions = df.groupby(['เกรดโครงการ','รูปร่างที่ดิน'])[[f'{h}_prop' for h in house_types]].mean()

# Finished setup — next phase would include prediction functions and UI integration

st.success("✅ ข้อมูลพร้อมแล้วสำหรับการพัฒนาโมเดลทำนาย")
# --- 3. Prediction Function (Internal, works with SQM) ---
# This function will work internally with SQM, and the main UI function will handle SQW conversion
def predict_project_layout_internal(
    project_area_sqm_input: float, # Input is expected in SQM here
    land_shape: str,
    project_grade: str,
    province: str # Province is currently not used in prediction logic but kept for future expansion
) -> dict:
    """
    Predicts various layout metrics for a new project based on historical data.
    All internal calculations are in Square Meters.
    """
    project_area_sqm = project_area_sqm_input # Use this variable for calculations

    # 1. Predict Area Allocations
    predicted_public_area = project_area_sqm * avg_public_area_ratio
    predicted_distributable_area = project_area_sqm * avg_distributable_area_ratio
    predicted_garden_area = predicted_distributable_area * 0.05 # Assuming 5% of distributable for garden
    predicted_road_area = project_area_sqm * avg_road_area_ratio

    # 2. Predict Number of Units by Type and Total Units
    predicted_units = {
        'ทาวโฮม': 0,
        'บ้านแฝด': 0,
        'บ้านเดี่ยว': 0,
        'บ้านเดี่ยว3ชั้น': 0,
        'อาคารพาณิชย์': 0
    }
    total_predicted_units = 0

    # Get proportions based on Grade and Land Shape
    proportions_key = (project_grade, land_shape)
    if proportions_key in grade_land_shape_proportions.index:
        proportions_series = grade_land_shape_proportions.loc[proportions_key]
        proportions_df = proportions_series.to_frame().T # Convert to DataFrame row for consistent access
    else:
        proportions_df = None # No specific historical data for this combination

    # Apply grade-specific rules first
    if project_grade in grade_rules:
        # For LUXURY, set all others to 0 and ensure บ้านเดี่ยว takes all proportion
        if project_grade == 'LUXURY':
            for h_type in house_types: # Ensure 'house_types' is the corrected list
                if h_type != 'บ้านเดี่ยว':
                    predicted_units[h_type] = 0
            remaining_house_types = ['บ้านเดี่ยว']
        else: # For other grades with rules (e.g., PREMIUM)
            for h_type, value in grade_rules[project_grade].items():
                predicted_units[h_type] = value
            remaining_house_types = [h for h in house_types if h not in grade_rules[project_grade]]
        
        # Calculate base units from distributable area
        estimated_total_units_from_area = 0
        if not np.isnan(avg_units_per_dist_area) and avg_units_per_dist_area > 0:
            estimated_total_units_from_area = round(predicted_distributable_area * avg_units_per_dist_area)

        # Distribute remaining units based on proportions
        if proportions_df is not None:
            current_proportions = proportions_df[[f'{h_type}_prop' for h_type in remaining_house_types]].values.flatten()
            if current_proportions.sum() > 0:
                current_proportions /= current_proportions.sum() # Normalize remaining proportions
            else: # Fallback to general average if normalized sum is zero (shouldn't happen if data exists for remaining types)
                general_avg_props_remaining = df[[f'{h_type}_prop' for h_type in remaining_house_types]].mean().values
                if general_avg_props_remaining.sum() > 0:
                    current_proportions = general_avg_props_remaining / general_avg_props_remaining.sum()
                else: # If still zero, distribute equally among remaining if any
                    if len(remaining_house_types) > 0:
                        current_proportions = np.ones(len(remaining_house_types)) / len(remaining_house_types)
                    else:
                        current_proportions = np.array([]) # No remaining types

        else: # Fallback to general average if no specific proportions found for grade/shape
            general_avg_props_remaining = df[[f'{h_type}_prop' for h_type in remaining_house_types]].mean().values
            if general_avg_props_remaining.sum() > 0:
                current_proportions = general_avg_props_remaining / general_avg_props_remaining.sum()
            else: # If still zero, distribute equally among remaining if any
                if len(remaining_house_types) > 0:
                    current_proportions = np.ones(len(remaining_house_types)) / len(remaining_house_types)
                else:
                    current_proportions = np.array([]) # No remaining types


        # Distribute estimated total units among remaining house types
        temp_predicted_remaining_units = {}
        for i, h_type in enumerate(remaining_house_types):
            if i < len(current_proportions): # Ensure index is within bounds
                temp_predicted_remaining_units[h_type] = round(estimated_total_units_from_area * current_proportions[i])
        
        # Merge back with already set units (e.g., those fixed at 0 by grade rules)
        for h_type in remaining_house_types:
            predicted_units[h_type] = temp_predicted_remaining_units.get(h_type, 0)


        # Refine units based on area consumed by primary types (TH, BA, BD)
        total_area_consumed_by_calculated_types = (predicted_units['ทาวโฮม'] * AREA_TH +
                                              predicted_units['บ้านแฝด'] * AREA_BA +
                                              predicted_units['บ้านเดี่ยว'] * AREA_BD)
        
        if total_area_consumed_by_calculated_types > 0 and predicted_distributable_area > 0:
            scale_factor = predicted_distributable_area / total_area_consumed_by_calculated_types
            for h_type in ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว']: # Only scale types with defined areas
                predicted_units[h_type] = round(predicted_units[h_type] * scale_factor)

    else: # No specific grade rules, use average proportions for the given grade/land shape
        # Calculate base units from distributable area
        estimated_total_units_from_area = 0
        if not np.isnan(avg_units_per_dist_area) and avg_units_per_dist_area > 0:
            estimated_total_units_from_area = round(predicted_distributable_area * avg_units_per_dist_area)
            
        if proportions_df is not None:
            # Use specific proportions if available
            for h_type in house_types:
                predicted_units[h_type] = round(estimated_total_units_from_area * proportions_df[f'{h_type}_prop'].iloc[0])
        else:
            # Fallback to general average proportions if no specific (grade, land_shape) combo found
            general_avg_props = df[[f'{h_type}_prop' for h_type in house_types]].mean()
            if general_avg_props.sum() > 0:
                general_avg_props /= general_avg_props.sum() # Normalize
            for h_type in house_types:
                predicted_units[h_type] = round(estimated_total_units_from_area * general_avg_props[f'{h_type}_prop'])
        
        # Refine units based on area consumed if possible
        total_area_consumed_by_known_types = (predicted_units['ทาวโฮม'] * AREA_TH +
                                              predicted_units['บ้านแฝด'] * AREA_BA +
                                              predicted_units['บ้านเดี่ยว'] * AREA_BD)
        
        if total_area_consumed_by_known_types > 0 and predicted_distributable_area > 0:
            scale_factor = predicted_distributable_area / total_area_consumed_by_known_types
            for h_type in ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว']:
                predicted_units[h_type] = round(predicted_units[h_type] * scale_factor)
                
    total_predicted_units = sum(predicted_units.values())

    # 3. Predict Number of Alleys (จำนวนซอย)
    predicted_alleys = 0
    if total_predicted_units > 0 and not np.isnan(avg_alley_per_unit):
        predicted_alleys = round(total_predicted_units * avg_alley_per_unit)
    elif predicted_distributable_area > 0 and not np.isnan(avg_alley_per_dist_area):
        predicted_alleys = round(predicted_distributable_area * avg_alley_per_dist_area)
    
    predicted_alleys = max(1, predicted_alleys) # At least one alley for a project.


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
# This part runs once when the app starts to show overall model accuracy
actual_total_units = []
predicted_total_units_for_metrics = []
actual_distributable_areas = [] # In SQM
predicted_distributable_areas_for_metrics = [] # In SQM
actual_alleys = []
predicted_alleys_for_metrics = []

for index, row in df.iterrows():
    # Make sure to pass project_area_sqm to the function, which will be converted inside
    # We are calculating metrics based on actual historical data that is in SQM
    
    # Check if necessary columns exist before proceeding for metrics calculation
    if 'พื้นที่โครงการ(ตรม)' in row and 'รูปร่างที่ดิน' in row and \
       'เกรดโครงการ' in row and 'จังหวัด' in row and \
       'จำนวนหลัง' in row and 'พื้นที่จัดจำหน่าย(ตรม)' in row and 'จำนวนซอย' in row:
        
        pred_metrics = predict_project_layout_internal( # Use internal function that expects SQM
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
        st.warning(f"ข้ามแถวที่ {index} ในข้อมูลย้อนหลังเนื่องจากข้อมูลไม่ครบถ้วนสำหรับการคำนวณ Metrics")


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
st.set_page_config(
    page_title="Layout Project Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🏡 การทำนายผังโครงการใหม่")
st.markdown("โปรดกรอกข้อมูลสำหรับโครงการใหม่เพื่อรับการทำนายผังและจำนวนบ้าน.")

# Display Model Performance Metrics in the sidebar
st.sidebar.header("📊 ประสิทธิภาพโมเดล (จากการทำนายข้อมูลย้อนหลัง)")
st.sidebar.metric(label="จำนวนแปลง (รวม) - ค่าคลาดเคลื่อนเฉลี่ย (MEP)", value=f"{metrics_total_units['MEP']:.2f}%")
st.sidebar.metric(label="จำนวนแปลง (รวม) - R-squared ($R^2$)", value=f"{metrics_total_units['R2']:.2f}")
st.sidebar.markdown("---")
st.sidebar.metric(label="พื้นที่จัดจำหน่าย - ค่าคลาดเคลื่อนเฉลี่ย (MEP)", value=f"{metrics_dist_area['MEP']:.2f}%")
st.sidebar.metric(label="พื้นที่จัดจำหน่าย - R-squared ($R^2$)", value=f"{metrics_dist_area['R2']:.2f}")
st.sidebar.markdown("---")
st.sidebar.metric(label="จำนวนซอย - ค่าคลาดเคลื่อนเฉลี่ย (MEP)", value=f"{metrics_alleys['MEP']:.2f}%")
st.sidebar.metric(label="จำนวนซอย - R-squared ($R^2$)", value=f"{metrics_alleys['R2']:.2f}")

# Get unique values for dropdowns from loaded data
# ตรวจสอบว่าคอลัมน์มีอยู่จริงก่อนที่จะดึงค่า unique
unique_land_shapes = df['รูปร่างที่ดิน'].unique().tolist() if 'รูปร่างที่ดิน' in df.columns else []
unique_grades = df['เกรดโครงการ'].unique().tolist() if 'เกรดโครงการ' in df.columns else []
unique_provinces = df['จังหวัด'].unique().tolist() if 'จังหวัด' in df.columns else []


# Input widgets
st.header("ข้อมูลโครงการใหม่")
col1, col2 = st.columns(2)
with col1:
    project_area_sqw = st.number_input(
        "พื้นที่โครงการ (ตรว.)", # Changed to Square Wah
        min_value=250.0, # Equivalent to 1000 sqm (1000/4)
        max_value=250000.0, # Equivalent to 1,000,000 sqm (1,000,000/4)
        value=12500.0, # Equivalent to 50,000 sqm (50,000/4)
        step=250.0, # Equivalent to 1000 sqm (1000/4)
        help="ป้อนพื้นที่รวมของโครงการใหม่เป็นตารางวา"
    )
with col2:
    if unique_land_shapes:
        land_shape = st.selectbox(
            "รูปร่างที่ดิน",
            options=unique_land_shapes,
            help="เลือกรูปร่างที่ดินของโครงการใหม่"
        )
    else:
        land_shape = st.selectbox(
            "รูปร่างที่ดิน",
            options=["ไม่พบข้อมูล"], # Fallback
            help="ไม่พบข้อมูลรูปร่างที่ดินในไฟล์ CSV"
        )
        st.warning("ไม่พบข้อมูลรูปร่างที่ดินในไฟล์ CSV กรุณาตรวจสอบคอลัมน์ 'รูปร่างที่ดิน'")


col3, col4 = st.columns(2)
with col3:
    if unique_grades:
        project_grade = st.selectbox(
            "เกรดโครงการ",
            options=unique_grades,
            help="เลือกเกรดของโครงการ (เช่น PREMIUM, LUXURY, BELLA)"
        )
    else:
        project_grade = st.selectbox(
            "เกรดโครงการ",
            options=["ไม่พบข้อมูล"], # Fallback
            help="ไม่พบข้อมูลเกรดโครงการในไฟล์ CSV"
        )
        st.warning("ไม่พบข้อมูลเกรดโครงการในไฟล์ CSV กรุณาตรวจสอบคอลัมน์ 'เกรดโครงการ'")

with col4:
    if unique_provinces:
        province = st.selectbox(
            "จังหวัด",
            options=unique_provinces,
            help="เลือกจังหวัดที่โครงการตั้งอยู่ (ปัจจุบันไม่ได้ใช้ในการคำนวณโดยตรง)"
        )
    else:
        province = st.selectbox(
            "จังหวัด",
            options=["ไม่พบข้อมูล"], # Fallback
            help="ไม่พบข้อมูลจังหวัดในไฟล์ CSV"
        )
        st.warning("ไม่พบข้อมูลจังหวัดในไฟล์ CSV กรุณาตรวจสอบคอลัมน์ 'จังหวัด'")


if st.button("ทำนายผังโครงการ"):
    # ตรวจสอบว่าเลือกข้อมูลจาก Dropdown ที่ถูกต้อง
    if "ไม่พบข้อมูล" in [land_shape, project_grade, province]:
        st.error("กรุณาแก้ไขข้อผิดพลาดเกี่ยวกับการไม่พบข้อมูลใน dropdowns ก่อนทำการทำนาย")
    elif project_area_sqw <= 0:
        st.error("กรุณาป้อน 'พื้นที่โครงการ' ที่มากกว่า 0.")
    else:
        # Perform prediction using the SQW function
        with st.spinner("กำลังคำนวณ..."):
            predicted_results = predict_project_layout_sqw(
                project_area_sqw_input=project_area_sqw,
                land_shape=land_shape,
                project_grade=project_grade,
                province=province
            )
        
        st.success("ทำนายผลสำเร็จ!")
        st.header("ผลการทำนาย")
        
        # Display results in an organized way
        results_df = pd.DataFrame(predicted_results.items(), columns=["ตัวชี้วัด", "ค่าที่ทำนาย"])
        st.dataframe(results_df, hide_index=True)

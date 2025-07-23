import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Load Data ---
@st.cache_data # Cache the data loading to improve performance
def load_data():
    try:
        df = pd.read_csv('layoutdata.xlsx - Sheet1.csv')
        # Ensure column names are stripped of whitespace for consistency
        df.columns = df.columns.str.strip()
        
        # Calculate Total Units if not already present
        if 'จำนวนหลัง' not in df.columns:
            df['จำนวนหลัง'] = df['ทาวโฮม'] + df['บ้านแฝด'] + df['บ้านเดี่ยว'] + df['บ้านเดี่ยว3ชั้น'] + df['อาคารพาณิชย์']
        
        # Calculate proportion for each house type
        house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']
        df['total_houses_for_prop'] = df['ทาวโฮม'] + df['บ้านแฝด'] + df['บ้านเดี่ยว'] + df['บ้านเดี่ยว3ชั้น'] + df['อาคารพาณิชย์']
        for h_type in house_types:
            df[f'{h_type}_prop'] = df[h_type] / df['total_houses_for_prop']
        df.fillna(0, inplace=True) # Fill NaN proportions with 0

        return df
    except FileNotFoundError:
        st.error("Error: 'layoutdata.xlsx - Sheet1.csv' not found. Please ensure the file is in the same directory as the script.")
        st.stop() # Stop the app if data is not found
    except Exception as e:
        st.error(f"An error occurred while loading or processing data: {e}")
        st.stop()

df = load_data()

# --- 2. Pre-calculate average ratios and proportions from historical data ---
# Ratios for area calculations
# Use original column names for robustness after stripping whitespace
if 'พื้นที่สาธา(ตรม)' in df.columns and 'พื้นที่โครงการ(ตรม)' in df.columns:
    avg_public_area_ratio = (df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_public_area_ratio = 0.333 # Fallback if column not found
    st.warning("Column 'พื้นที่สาธา(ตรม)' or 'พื้นที่โครงการ(ตรม)' not found. Using default public area ratio.")

if 'พื้นที่จัดจำหน่าย(ตรม)' in df.columns and 'พื้นที่โครงการ(ตรม)' in df.columns:
    avg_distributable_area_ratio = (df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_distributable_area_ratio = 0.667 # Fallback
    st.warning("Column 'พื้นที่จัดจำหน่าย(ตรม)' or 'พื้นที่โครงการ(ตรม)' not found. Using default distributable area ratio.")

if 'พื้นที่ถนนรวม' in df.columns and 'พื้นที่โครงการ(ตรม)' in df.columns:
    avg_road_area_ratio = (df['พื้นที่ถนนรวม'] / df['พื้นที่โครงการ(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_road_area_ratio = 0.30 # Fallback
    st.warning("Column 'พื้นที่ถนนรวม' or 'พื้นที่โครงการ(ตรม)' not found. Using default road area ratio.")


# Average area per unit for each type (from user's request)
AREA_TH = 5 * 16  # ทาวน์โฮม
AREA_BA = 12 * 16 # บ้านแฝด
AREA_BD = 15 * 18 # บ้านเดี่ยว

# Average units per distributable area (overall)
if 'จำนวนหลัง' in df.columns and 'พื้นที่จัดจำหน่าย(ตรม)' in df.columns:
    avg_units_per_dist_area = (df['จำนวนหลัง'] / df['พื้นที่จัดจำหน่าย(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_units_per_dist_area = 0.005 # Fallback (e.g. 1 unit per 200 sqm)
    st.warning("Column 'จำนวนหลัง' or 'พื้นที่จัดจำหน่าย(ตรม)' not found. Using default units per distributable area.")

# House types list for iteration
house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']

# Group by 'เกรดโครงการ' and 'รูปร่างที่ดิน' to get average proportions
grade_land_shape_proportions = df.groupby(['เกรดโครงการ', 'รูปร่างที่ดิน'])[
    [f'{h_type}_prop' for h_type in house_types]
].mean()

# Rules for specific grades (based on initial observation)
grade_rules = {
    'LUXURY': {'ทาวโฮม': 0, 'บ้านแฝด': 0, 'บ้านเดี่ยว3ชั้น': 0, 'อาคารพาณิชย์': 0}, # Only บ้านเดี่ยว
    'PREMIUM': {'ทาวโฮม': 0, 'บ้านเดี่ยว3ชั้น': 0, 'อาคารพาณิชย์': 0} # Mostly บ้านเดี่ยว, some บ้านแฝด possible
    # Add other grade rules if identified
}

# Average number of alleys per total units
if 'จำนวนซอย' in df.columns and 'จำนวนหลัง' in df.columns:
    avg_alley_per_unit = (df['จำนวนซอย'] / df['จำนวนหลัง']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_alley_per_unit = 0.05 # Fallback (e.g. 1 alley per 20 units)
    st.warning("Column 'จำนวนซอย' or 'จำนวนหลัง' not found. Using default alleys per unit.")

if 'จำนวนซอย' in df.columns and 'พื้นที่จัดจำหน่าย(ตรม)' in df.columns:
    avg_alley_per_dist_area = (df['จำนวนซอย'] / df['พื้นที่จัดจำหน่าย(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_alley_per_dist_area = 0.0001 # Fallback (e.g. 1 alley per 10000 sqm)
    st.warning("Column 'จำนวนซอย' or 'พื้นที่จัดจำหน่าย(ตรม)' not found. Using default alleys per distributable area.")

# --- 3. Prediction Function ---
def predict_project_layout(
    project_area_sqm: float,
    land_shape: str,
    project_grade: str,
    province: str # Province is currently not used in prediction logic but kept for future expansion
) -> dict:
    """
    Predicts various layout metrics for a new project based on historical data.
    """

    # 1. Predict Area Allocations
    predicted_public_area = project_area_sqm * avg_public_area_ratio
    predicted_distributable_area = project_area_sqm * avg_distributable_area_ratio
    predicted_garden_area = predicted_distributable_area * 0.05
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
        for h_type, value in grade_rules[project_grade].items():
            predicted_units[h_type] = value
        
        remaining_house_types = [h for h in house_types if h not in grade_rules[project_grade]]
        
        # Calculate base units from distributable area, if avg_units_per_dist_area is valid
        estimated_total_units_from_area = 0
        if not np.isnan(avg_units_per_dist_area) and avg_units_per_dist_area > 0:
            estimated_total_units_from_area = round(predicted_distributable_area * avg_units_per_dist_area)

        if proportions_df is not None:
            current_proportions = proportions_df[[f'{h_type}_prop' for h_type in remaining_house_types]].values.flatten()
            if current_proportions.sum() > 0:
                current_proportions /= current_proportions.sum() # Normalize remaining proportions
            else: # Fallback to general average if normalized sum is zero
                current_proportions = df[[f'{h_type}_prop' for h_type in remaining_house_types]].mean().values
                if current_proportions.sum() > 0:
                    current_proportions /= current_proportions.sum()
        else: # Fallback to general average if no specific proportions found for grade/shape
            current_proportions = df[[f'{h_type}_prop' for h_type in remaining_house_types]].mean().values
            if current_proportions.sum() > 0:
                current_proportions /= current_proportions.sum()

        # Distribute estimated total units among remaining house types
        for i, h_type in enumerate(remaining_house_types):
            if i < len(current_proportions): # Ensure index is within bounds
                predicted_units[h_type] = round(estimated_total_units_from_area * current_proportions[i])

        # Refine units based on area consumed
        total_area_consumed_by_known_types = (predicted_units['ทาวโฮม'] * AREA_TH +
                                              predicted_units['บ้านแฝด'] * AREA_BA +
                                              predicted_units['บ้านเดี่ยว'] * AREA_BD)
        
        if total_area_consumed_by_known_types > 0 and predicted_distributable_area > 0:
            scale_factor = predicted_distributable_area / total_area_consumed_by_known_types
            for h_type in ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว']:
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

# --- 4. Streamlit UI ---
st.set_page_config(
    page_title="Layout Project Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🏡 การทำนายผังโครงการใหม่")
st.markdown("โปรดกรอกข้อมูลสำหรับโครงการใหม่เพื่อรับการทำนายผังและจำนวนบ้าน.")

# Get unique values for dropdowns from loaded data
unique_land_shapes = df['รูปร่างที่ดิน'].unique().tolist()
unique_grades = df['เกรดโครงการ'].unique().tolist()
unique_provinces = df['จังหวัด'].unique().tolist()

# Input widgets
st.header("ข้อมูลโครงการใหม่")
col1, col2 = st.columns(2)

with col1:
    project_area = st.number_input(
        "พื้นที่โครงการ (ตรม.)",
        min_value=1000.0,
        max_value=1000000.0,
        value=50000.0,
        step=1000.0,
        help="ป้อนพื้นที่รวมของโครงการใหม่เป็นตารางเมตร"
    )

with col2:
    land_shape = st.selectbox(
        "รูปร่างที่ดิน",
        options=unique_land_shapes,
        help="เลือกรูปร่างที่ดินของโครงการใหม่"
    )

col3, col4 = st.columns(2)
with col3:
    project_grade = st.selectbox(
        "เกรดโครงการ",
        options=unique_grades,
        help="เลือกเกรดของโครงการ (เช่น PREMIUM, LUXURY, BELLA)"
    )

with col4:
    province = st.selectbox(
        "จังหวัด",
        options=unique_provinces,
        help="เลือกจังหวัดที่โครงการตั้งอยู่ (ปัจจุบันไม่ได้ใช้ในการคำนวณโดยตรง)"
    )

if st.button("ทำนายผังโครงการ"):
    if project_area <= 0:
        st.error("กรุณาป้อน 'พื้นที่โครงการ' ที่มากกว่า 0.")
    else:
        # Perform prediction
        with st.spinner("กำลังคำนวณ..."):
            predicted_results = predict_project_layout(
                project_area_sqm=project_area,
                land_shape=land_shape,
                project_grade=project_grade,
                province=province
            )
        
        st.success("ทำนายผลสำเร็จ!")
        st.header("ผลการทำนาย")
        
        # Display results in an organized way (e.g., using st.metric or a DataFrame)
        results_df = pd.DataFrame(predicted_results.items(), columns=["ตัวชี้วัด", "ค่าที่ทำนาย"])
        st.dataframe(results_df, hide_index=True)

        st.markdown("""
        ---
        **หมายเหตุ:**
        * การทำนายนี้ใช้ค่าเฉลี่ยและสัดส่วนจากข้อมูลในอดีต.
        * สำหรับบ้านเดี่ยว 3 ชั้น และอาคารพาณิชย์ ไม่ได้มีการกำหนดพื้นที่มาตรฐานต่อหลัง ทำให้การคำนวณจำนวนแปลงอาจไม่แม่นยำเท่าทาวน์โฮม บ้านแฝด และบ้านเดี่ยว.
        * 'จำนวนแปลง (รวม)' คือผลรวมของบ้านทุกประเภท.
        * 'จำนวนซอย' มีค่าต่ำสุดที่ 1.
        """)

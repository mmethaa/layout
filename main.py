import streamlit as st
import pandas as pd
import numpy as np

# Initialize df outside try block to ensure it always exists, even if None
df = None

# --- 1. Load Data ---
# Load the historical data directly from the file system.
# The file 'layoutdata.xlsx - Sheet1.csv' MUST be in the same directory as this script.
try:
    # IMPORTANT: Ensure 'layoutdata.xlsx - Sheet1.csv' is a true CSV file.
    # If it was originally an Excel file, open it in Excel and 'Save As' -> 'CSV (Comma delimited)'.
    # Added encoding='utf-8' for robustness.
    df = pd.read_csv('layoutdata.xlsx - Sheet1.csv', encoding='utf-8')
    
    # Ensure column names are stripped of whitespace for consistency
    df.columns = df.columns.str.strip()
    
    # Calculate Total Units if not already present or needs recalculation based on specific columns
    required_house_cols = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']
    existing_house_cols = [col for col in required_house_cols if col in df.columns]

    if existing_house_cols:
        df['จำนวนหลัง'] = df[existing_house_cols].sum(axis=1)
    else:
        # Fallback if house type columns are missing
        st.error("บางคอลัมน์ประเภทบ้าน (ทาวโฮม, บ้านแฝด, ฯลฯ) ขาดหายไปในไฟล์ข้อมูล. ไม่สามารถคำนวณ 'จำนวนหลัง' ได้อย่างแม่นยำ.")
        df['จำนวนหลัง'] = 0 # Default to 0, might lead to less accurate predictions

    # Calculate proportion for each house type
    if existing_house_cols:
        df['total_houses_for_prop'] = df[existing_house_cols].sum(axis=1)
        # Handle division by zero for projects with no houses
        for h_type in existing_house_cols: # Loop through existing house types
             df[f'{h_type}_prop'] = df[h_type] / df['total_houses_for_prop'].replace(0, np.nan) 
        df.fillna(0, inplace=True) # Fill NaN proportions with 0
    else:
        st.warning("ไม่พบคอลัมน์ประเภทบ้านสำหรับคำนวณสัดส่วนในไฟล์ข้อมูล.")

    # Check for essential columns for ratios and dropdowns
    essential_cols = ['พื้นที่สาธา(ตรม)', 'พื้นที่โครงการ(ตรม)', 'พื้นที่จัดจำหน่าย(ตรม)', 
                      'พื้นที่ถนนรวม', 'จำนวนซอย', 'รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด']
    for col in essential_cols:
        if col not in df.columns:
            st.warning(f"คำเตือน: คอลัมน์ '{col}' ไม่พบในไฟล์ข้อมูล. การทำนายบางส่วนอาจไม่แม่นยำ.")

except FileNotFoundError:
    st.error("Error: ไม่พบไฟล์ 'layoutdata.xlsx - Sheet1.csv' โปรดตรวจสอบให้แน่ใจว่าไฟล์อยู่ในไดเรกทอรีเดียวกันกับสคริปต์.")
    st.stop() # Stop the app if data is not found
except Exception as e: # This will catch the 'str' object has no attribute 'xlsx' error
    st.error(f"เกิดข้อผิดพลาดในการโหลดหรือประมวลผลไฟล์: {e}")
    st.stop()

# --- 2. Pre-calculate average ratios and proportions from historical data ---
# Ratios for area calculations
avg_public_area_ratio = (df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']).replace([np.inf, -np.inf], np.nan).mean() if df is not None and 'พื้นที่สาธา(ตรม)' in df.columns and 'พื้นที่โครงการ(ตรม)' in df.columns else 0.333
avg_distributable_area_ratio = (df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']).replace([np.inf, -np.inf], np.nan).mean() if df is not None and 'พื้นที่จัดจำหน่าย(ตรม)' in df.columns and 'พื้นที่โครงการ(ตรม)' in df.columns else 0.667
avg_road_area_ratio = (df['พื้นที่ถนนรวม'] / df['พื้นที่โครงการ(ตรม)']).replace([np.inf, -np.inf], np.nan).mean() if df is not None and 'พื้นที่ถนนรวม' in df.columns and 'พื้นที่โครงการ(ตรม)' in df.columns else 0.30

# Average area per unit for each type (from user's request)
AREA_TH = 5 * 16  # ทาวน์โฮม (ตรม.)
AREA_BA = 12 * 16 # บ้านแฝด (ตรม.)
AREA_BD = 15 * 18 # บ้านเดี่ยว (ตรม.)

# Average units per distributable area (overall)
if df is not None and 'จำนวนหลัง' in df.columns and 'พื้นที่จัดจำหน่าย(ตรม)' in df.columns:
    avg_units_per_dist_area = (df['จำนวนหลัง'] / df['พื้นที่จัดจำหน่าย(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_units_per_dist_area = 0.005 # Fallback value if columns missing or data problematic

# House types list for iteration
house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']

# Group by 'เกรดโครงการ' and 'รูปร่างที่ดิน' to get average proportions
# Filter for only existing proportion columns before grouping
existing_prop_cols = [f'{h_type}_prop' for h_type in house_types if f'{h_type}_prop' in df.columns] if df is not None else []
if df is not None and 'เกรดโครงการ' in df.columns and 'รูปร่างที่ดิน' in df.columns and existing_prop_cols:
    grade_land_shape_proportions = df.groupby(['เกรดโครงการ', 'รูปร่างที่ดิน'])[existing_prop_cols].mean()
else:
    grade_land_shape_proportions = pd.DataFrame() # Empty if not enough data

# Rules for specific grades (based on initial observation)
grade_rules = {
    'LUXURY': {'ทาวโฮม': 0, 'บ้านแฝด': 0, 'บ้านเดี่ยว3ชั้น': 0, 'อาคารพาณิชย์': 0}, # Only บ้านเดี่ยว
    'PREMIUM': {'ทาวโฮม': 0, 'บ้านเดี่ยว3ชั้น': 0, 'อาคารพาณิชย์': 0} # Mostly บ้านเดี่ยว, some บ้านแฝด possible
    # Add other grade rules if identified
}

# Average number of alleys per total units
if df is not None and 'จำนวนซอย' in df.columns and 'จำนวนหลัง' in df.columns:
    avg_alley_per_unit = (df['จำนวนซอย'] / df['จำนวนหลัง']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_alley_per_unit = 0.05 # Fallback (e.g. 1 alley per 20 units)

if df is not None and 'จำนวนซอย' in df.columns and 'พื้นที่จัดจำหน่าย(ตรม)' in df.columns:
    avg_alley_per_dist_area = (df['จำนวนซอย'] / df['พื้นที่จัดจำหน่าย(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()
else:
    avg_alley_per_dist_area = 0.0001 # Fallback (e.g. 1 alley per 10000 sqm)


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
    predicted_units = {h_type: 0 for h_type in house_types}
    total_predicted_units = 0

    # Get proportions based on Grade and Land Shape
    proportions_key = (project_grade, land_shape)
    proportions_series = None
    if proportions_key in grade_land_shape_proportions.index:
        proportions_series = grade_land_shape_proportions.loc[proportions_key]
    
    # Calculate base units from distributable area, if avg_units_per_dist_area is valid
    estimated_total_units_from_area = 0
    if not np.isnan(avg_units_per_dist_area) and avg_units_per_dist_area > 0:
        estimated_total_units_from_area = round(predicted_distributable_area * avg_units_per_dist_area)
    
    # Apply grade-specific rules first
    if project_grade in grade_rules:
        for h_type, value in grade_rules[project_grade].items():
            predicted_units[h_type] = value
        
        remaining_house_types = [h for h in house_types if h not in grade_rules[project_grade]]
        
        current_proportions = None
        if proportions_series is not None:
            # Filter proportions series to only include existing house types and those not set to 0 by grade rules
            current_proportions_temp_dict = {h: proportions_series[f'{h}_prop'] for h in remaining_house_types if f'{h}_prop' in proportions_series.index}
            current_proportions_temp = pd.Series(current_proportions_temp_dict)
            
            if current_proportions_temp.sum() > 0:
                current_proportions = current_proportions_temp / current_proportions_temp.sum()
            
        if current_proportions is None or current_proportions.empty: # Fallback to general average if specific or normalized sum is zero or no data
            general_props_temp_dict = {h: df[f'{h}_prop'].mean() for h in remaining_house_types if f'{h}_prop' in df.columns} if df is not None else {}
            general_props_temp = pd.Series(general_props_temp_dict)

            if general_props_temp.sum() > 0:
                current_proportions = general_props_temp / general_props_temp.sum()
            elif remaining_house_types: # Even distribution if all else fails and there are types
                current_proportions = pd.Series([1/len(remaining_house_types)] * len(remaining_house_types), index=remaining_house_types) # Use type names directly
            else:
                current_proportions = pd.Series() # Empty series if no types to distribute

        # Distribute estimated total units among remaining house types
        if current_proportions is not None and not current_proportions.empty:
            for h_type, prop_val in current_proportions.items():
                predicted_units[h_type] = round(estimated_total_units_from_area * prop_val)

        # Refine units based on area consumed
        total_area_consumed_by_known_types = (predicted_units['ทาวโฮม'] * AREA_TH +
                                              predicted_units['บ้านแฝด'] * AREA_BA +
                                              predicted_units['บ้านเดี่ยว'] * AREA_BD)
        
        if total_area_consumed_by_known_types > 0 and predicted_distributable_area > 0:
            scale_factor = predicted_distributable_area / total_area_consumed_by_known_types
            for h_type in ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว']: # Only types with known area
                predicted_units[h_type] = max(0, round(predicted_units[h_type] * scale_factor))

    else: # No specific grade rules, use average proportions for the given grade/land shape
        # Calculate base units from distributable area
        estimated_total_units_from_area = 0
        if not np.isnan(avg_units_per_dist_area) and avg_units_per_dist_area > 0:
            estimated_total_units_from_area = round(predicted_distributable_area * avg_units_per_dist_area)
            
        current_proportions = None
        if proportions_series is not None:
            current_proportions_temp_dict = {h: proportions_series[f'{h}_prop'] for h in house_types if f'{h}_prop' in proportions_series.index}
            current_proportions_temp = pd.Series(current_proportions_temp_dict)
            if current_proportions_temp.sum() > 0:
                current_proportions = current_proportions_temp / current_proportions_temp.sum()
        
        if current_proportions is None or current_proportions.empty: # Fallback to general average
            general_props_temp_dict = {h: df[f'{h}_prop'].mean() for h in house_types if f'{h}_prop' in df.columns} if df is not None else {}
            general_props_temp = pd.Series(general_props_temp_dict)

            if general_props_temp.sum() > 0:
                current_proportions = general_props_temp / general_props_temp.sum()
            elif house_types: # Even distribution if all else fails
                current_proportions = pd.Series([1/len(house_types)] * len(house_types), index=house_types)
            else:
                current_proportions = pd.Series()

        if current_proportions is not None and not current_proportions.empty:
            for h_type, prop_val in current_proportions.items():
                predicted_units[h_type] = round(estimated_total_units_from_area * prop_val)
        
        # Refine units based on area consumed if possible
        total_area_consumed_by_known_types = (predicted_units['ทาวโฮม'] * AREA_TH +
                                              predicted_units['บ้านแฝด'] * AREA_BA +
                                              predicted_units['บ้านเดี่ยว'] * AREA_BD)
        
        if total_area_consumed_by_known_types > 0 and predicted_distributable_area > 0:
            scale_factor = predicted_distributable_area / total_area_consumed_by_known_types
            for h_type in ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว']:
                predicted_units[h_type] = max(0, round(predicted_units[h_type] * scale_factor))
                
    total_predicted_units = sum(predicted_units.values())

    # 3. Predict Number of Alleys (จำนวนซอย)
    predicted_alleys = 0
    if total_predicted_units > 0 and not np.isnan(avg_alley_per_unit) and avg_alley_per_unit is not None:
        predicted_alleys = round(total_predicted_units * avg_alley_per_unit)
    elif predicted_distributable_area > 0 and not np.isnan(avg_alley_per_dist_area) and avg_alley_per_dist_area is not None:
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
st.markdown("โปรดกรอกข้อมูลสำหรับโครงการใหม่เพื่อรับการทำนายผังและจำนวนบ้าน. (ไฟล์ข้อมูล `layoutdata.xlsx - Sheet1.csv` ต้องอยู่ในไดเรกทอรีเดียวกับสคริปต์นี้)")

# Get unique values for dropdowns from loaded data
# Ensure columns exist before accessing unique values, provide fallback if not
unique_land_shapes = df['รูปร่างที่ดิน'].unique().tolist() if df is not None and 'รูปร่างที่ดิน' in df.columns else []
unique_grades = df['เกรดโครงการ'].unique().tolist() if df is not None and 'เกรดโครงการ' in df.columns else []
unique_provinces = df['จังหวัด'].unique().tolist() if df is not None and 'จังหวัด' in df.columns else []

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
    if unique_land_shapes:
        land_shape = st.selectbox(
            "รูปร่างที่ดิน",
            options=unique_land_shapes,
            help="เลือกรูปร่างที่ดินของโครงการใหม่"
        )
    else:
        st.error("ไม่พบข้อมูล 'รูปร่างที่ดิน' ในไฟล์ข้อมูล. โปรดตรวจสอบคอลัมน์นี้.")
        land_shape = "" # Default to empty if no data

col3, col4 = st.columns(2)
with col3:
    if unique_grades:
        project_grade = st.selectbox(
            "เกรดโครงการ",
            options=unique_grades,
            help="เลือกเกรดของโครงการ (เช่น PREMIUM, LUXURY, BELLA)"
        )
    else:
        st.error("ไม่พบข้อมูล 'เกรดโครงการ' ในไฟล์ข้อมูล. โปรดตรวจสอบคอลัมน์นี้.")
        project_grade = "" # Default to empty if no data

with col4:
    if unique_provinces:
        province = st.selectbox(
            "จังหวัด",
            options=unique_provinces,
            help="เลือกจังหวัดที่โครงการตั้งอยู่ (ปัจจุบันไม่ได้ใช้ในการคำนวณโดยตรง)"
        )
    else:
        st.warning("ไม่พบข้อมูล 'จังหวัด' ในไฟล์ข้อมูล. ช่องนี้จะไม่มีผลต่อการคำนวณ.")
        province = "N/A" # Default to N/A if no data

if st.button("ทำนายผังโครงการ"):
    # Basic validation for essential inputs
    if project_area <= 0:
        st.error("กรุณาป้อน 'พื้นที่โครงการ' ที่มากกว่า 0.")
    elif not land_shape: # Check if land_shape is empty string due to missing column
        st.error("ไม่สามารถทำนายได้: ไม่มีข้อมูล 'รูปร่างที่ดิน' ให้เลือก. โปรดตรวจสอบไฟล์ข้อมูลของคุณ.")
    elif not project_grade: # Check if project_grade is empty string due to missing column
        st.error("ไม่สามารถทำนายได้: ไม่มีข้อมูล 'เกรดโครงการ' ให้เลือก. โปรดตรวจสอบไฟล์ข้อมูลของคุณ.")
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

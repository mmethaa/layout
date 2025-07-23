import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score # Import for metrics

# --- 1. Load Data ---
@st.cache_data # Cache the data loading to improve performance
def load_data():
    try:
        df = pd.read_excel("layoutdata.xlsx", sheet_name=sheet_name)
        # Ensure column names are stripped of whitespace for consistency
        df.columns = df.columns.str.strip()
        
        # Calculate Total Units if not already present
        if 'จำนวนหลัง' not in df.columns:
            df['จำนวนหลัง'] = df['ทาวโฮม'] + df['บ้านแฝด'] + df['บ้านเดี่ยว'] + df['บ้านเดี่ยว3ชั้น'] + df['อาคารพาณิชย์']
        
        # Calculate proportion for each house type
        house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิณย์']
        df['total_houses_for_prop'] = df['ทาวโฮม'] + df['บ้านแฝด'] + df['บ้านเดี่ยว'] + df['บ้านเดี่ยว3ชั้น'] + df['อาคารพาณิชย์']
        for h_type in house_types:
            # Handle division by zero for proportions
            df[f'{h_type}_prop'] = df.apply(lambda row: row[h_type] / row['total_houses_for_prop'] if row['total_houses_for_prop'] > 0 else 0, axis=1)
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

# Average area per unit for each type (from user's request - these are in SQM)
AREA_TH = 5 * 16  # ทาวน์โฮม (80 sqm)
AREA_BA = 12 * 16 # บ้านแฝด (192 sqm)
AREA_BD = 15 * 18 # บ้านเดี่ยว (270 sqm)

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

# Rules for specific grades (based on initial observation and user request)
# Ensure LUXURY only has บ้านเดี่ยว. PREMIUM mostly บ้านเดี่ยว and some บ้านแฝด.
grade_rules = {
    'LUXURY': {'ทาวโฮม': 0, 'บ้านแฝด': 0, 'บ้านเดี่ยว3ชั้น': 0, 'อาคารพาณิชย์': 0}, # Explicitly sets others to 0
    'PREMIUM': {'ทาวโฮม': 0, 'บ้านเดี่ยว3ชั้น': 0, 'อาคารพาณิชย์': 0} # Allows บ้านแฝด and บ้านเดี่ยว
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

# --- Helper function for metrics ---
def calculate_metrics(actual_values, predicted_values):
    """Calculates MEP and R-squared, handling potential NaNs and zero actuals for MEP."""
    actual_values = np.array(actual_values).flatten()
    predicted_values = np.array(predicted_values).flatten()

    # Filter out NaN values from both arrays to avoid errors
    valid_indices = ~np.isnan(actual_values) & ~np.isnan(predicted_values)
    actual_values = actual_values[valid_indices]
    predicted_values = predicted_values[valid_indices]

    if len(actual_values) == 0 or np.all(actual_values == 0) and np.all(predicted_values == 0):
        # If no valid data or all zeros and predictions are also all zeros, R2 is 1 (perfect fit for trivial case)
        # MEP is 0 if all are zeros.
        return {'MEP': 0.0, 'R2': 1.0}
    elif np.all(actual_values == 0): # All actuals are zero, but predictions are not all zero
        mep = np.mean(np.abs(predicted_values)) * 100 # Average absolute prediction when actual is 0
        r2 = r2_score(actual_values, predicted_values) # R2 will be < 0 or low
        return {'MEP': mep, 'R2': r2}

    # Mean Error Percentage (MEP)
    # Handle division by zero for actual_values if they can be 0 for specific entries
    diff_abs_percent = []
    for i in range(len(actual_values)):
        if actual_values[i] != 0:
            diff_abs_percent.append(np.abs((actual_values[i] - predicted_values[i]) / actual_values[i]))
        elif predicted_values[i] != 0: # Actual is 0 but predicted is not 0, contributes to error
            diff_abs_percent.append(1.0) # 100% error relative to 0 actual
        # If both are 0, it's 0% error, doesn't contribute to diff_abs_percent

    mep = np.mean(diff_abs_percent) * 100 if diff_abs_percent else 0.0

    # R-squared
    r2 = r2_score(actual_values, predicted_values)

    return {'MEP': mep, 'R2': r2}


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

metrics_total_units = calculate_metrics(actual_total_units, predicted_total_units_for_metrics)
metrics_dist_area = calculate_metrics(actual_distributable_areas, predicted_distributable_areas_for_metrics)
metrics_alleys = calculate_metrics(actual_alleys, predicted_alleys_for_metrics)


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
            for h_type in house_types:
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
            else: # Fallback to general average if normalized sum is zero (shouldn't happen if data exists)
                current_proportions = df[[f'{h_type}_prop' for h_type in remaining_house_types]].mean().values
                if current_proportions.sum() > 0:
                    current_proportions /= current_proportions.sum()
        else: # Fallback to general average if no specific proportions found for grade/shape
            current_proportions = df[[f'{h_type}_prop' for h_type in remaining_house_types]].mean().values
            if current_proportions.sum() > 0:
                current_proportions /= current_proportions.sum()
            else: # If still zero, distribute equally among remaining if any
                if len(remaining_house_types) > 0:
                    current_proportions = np.ones(len(remaining_house_types)) / len(remaining_house_types)


        # Distribute estimated total units among remaining house types
        temp_predicted_remaining_units = {}
        for i, h_type in enumerate(remaining_house_types):
            if i < len(current_proportions):
                temp_predicted_remaining_units[h_type] = round(estimated_total_units_from_area * current_proportions[i])
        
        # Merge back with already set units (e.g., those fixed at 0)
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
unique_land_shapes = df['รูปร่างที่ดิน'].unique().tolist()
unique_grades = df['เกรดโครงการ'].unique().tolist()
unique_provinces = df['จังหวัด'].unique().tolist()

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
    if project_area_sqw <= 0:
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

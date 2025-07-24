import streamlit as st
import pandas as pd
import numpy as np

# Load the historical data
try:
    df = pd.read_csv('layoutdata.xlsx - Sheet1.csv')
except FileNotFoundError:
    print("Error: 'layoutdata.xlsx - Sheet1.csv' not found. Please ensure the file is in the correct directory.")
    # Exit or handle the error appropriately
    exit()

# --- Pre-calculate average ratios and proportions from historical data ---

# Ratios for area calculations
avg_public_area_ratio = (df['พื้นที่สาธา(ตรม)'] / df['พื้นที่โครงการ(ตรม)']).mean()
avg_distributable_area_ratio = (df['พื้นที่จัดจำหน่าย(ตรม)'] / df['พื้นที่โครงการ(ตรม)']).mean()
avg_road_area_ratio = (df['พื้นที่ถนนรวม'] / df['พื้นที่โครงการ(ตรม)']).mean()

# Average area per unit for each type (from user's request)
AREA_TH = 5 * 16  # ทาวน์โฮม
AREA_BA = 12 * 16 # บ้านแฝด
AREA_BD = 15 * 18 # บ้านเดี่ยว

# Average units per distributable area (overall)
# Calculate total units
df['Total_Units'] = df['ทาวโฮม'] + df['บ้านแฝด'] + df['บ้านเดี่ยว'] + df['บ้านเดี่ยว3ชั้น'] + df['อาคารพาณิชย์']
# Avoid division by zero for projects with no distributable area or no units
avg_units_per_dist_area = (df['Total_Units'] / df['พื้นที่จัดจำหน่าย(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()


# Proportions of house types by Grade and Land Shape
# Create a combined 'type' column for easier calculation of proportions
df['total_houses'] = df['ทาวโฮม'] + df['บ้านแฝด'] + df['บ้านเดี่ยว'] + df['บ้านเดี่ยว3ชั้น'] + df['อาคารพาณิชย์']

# Calculate proportion for each house type
house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชมย์']
for h_type in house_types:
    df[f'{h_type}_prop'] = df[h_type] / df['total_houses']
df.fillna(0, inplace=True) # Fill NaN proportions with 0 (for projects with no specific house type)

# Group by 'เกรดโครงการ' and 'รูปร่างที่ดิน' to get average proportions
grade_land_shape_proportions = df.groupby(['เกรดโครงการ', 'รูปร่างที่ดิน'])[
    [f'{h_type}_prop' for h_type in house_types]
].mean()

# Rules for specific grades (based on initial observation)
grade_rules = {
    'LUXURY': {'ทาวโฮม': 0, 'บ้านแฝด': 0, 'บ้านเดี่ยว3ชั้น': 0, 'อาคารพาณิชย์': 0}, # Only บ้านเดี่ยว
    'PREMIUM': {'ทาวโฮม': 0, 'บ้านเดี่ยว3ชั้น': 0, 'อาคารพาณิชย์': 0} # Mostly บ้านเดี่ยว, some บ้านแฝด possible
    # Other grades will use average proportions
}

# Average number of alleys per total units
avg_alley_per_unit = (df['จำนวนซอย'] / df['Total_Units']).replace([np.inf, -np.inf], np.nan).mean()
# Fallback if units is zero
avg_alley_per_dist_area = (df['จำนวนซอย'] / df['พื้นที่จัดจำหน่าย(ตรม)']).replace([np.inf, -np.inf], np.nan).mean()


def predict_project_layout(
    project_area_sqm: float,
    land_shape: str,
    project_grade: str,
    province: str
) -> dict:
    """
    Predicts various layout metrics for a new project based on historical data.

    Args:
        project_area_sqm (float): Total project area in square meters.
        land_shape (str): The shape of the land (e.g., 'รูปทรงสี่เหลี่ยม / บล็อกแน่น').
                          Valid values are from the 'รูปร่างที่ดิน' column in the dataset.
        project_grade (str): The grade of the project (e.g., 'BELLA', 'LUXURY').
                             Valid values are from the 'เกรดโครงการ' column.
        province (str): The province where the project is located.

    Returns:
        dict: A dictionary containing predicted layout metrics.
    """

    # 1. Predict Area Allocations
    predicted_public_area = project_area_sqm * avg_public_area_ratio
    predicted_distributable_area = project_area_sqm * avg_distributable_area_ratio
    predicted_garden_area = predicted_distributable_area * 0.05
    predicted_road_area = project_area_sqm * avg_road_area_ratio

    # Ensure total allocated area does not exceed project area
    total_allocated_sum = predicted_public_area + predicted_distributable_area + predicted_road_area
    if total_allocated_sum > project_area_sqm:
        # Adjust proportionally if sum exceeds total (shouldn't happen with ratios from same sum)
        # This is a fallback, ideally ratios sum to 1
        scale_factor = project_area_sqm / total_allocated_sum
        predicted_public_area *= scale_factor
        predicted_distributable_area *= scale_factor
        predicted_road_area *= scale_factor

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
    proportions_df = grade_land_shape_proportions.loc[(project_grade, land_shape), :] if (project_grade, land_shape) in grade_land_shape_proportions.index else None

    # Apply grade-specific rules first
    if project_grade in grade_rules:
        # Initialize with rules
        for h_type, value in grade_rules[project_grade].items():
            predicted_units[h_type] = value
        
        # Adjust proportions for remaining types
        remaining_house_types = [h for h in house_types if h not in grade_rules[project_grade]]
        
        if proportions_df is not None:
            # Use specific proportions if available
            current_proportions = proportions_df[
                [f'{h_type}_prop' for h_type in remaining_house_types]
            ].values.flatten()
            
            # Normalize remaining proportions
            if current_proportions.sum() > 0:
                current_proportions /= current_proportions.sum()
            else: # Fallback to general average if no historical data for remaining types for this grade/shape
                general_proportions = df[[f'{h_type}_prop' for h_type in remaining_house_types]].mean().values
                if general_proportions.sum() > 0:
                    general_proportions /= general_proportions.sum()
                current_proportions = general_proportions
        else:
            # Fallback to general average if no specific proportions found for grade/shape
            current_proportions = df[[f'{h_type}_prop' for h_type in remaining_house_types]].mean().values
            if current_proportions.sum() > 0:
                current_proportions /= current_proportions.sum()

        # Calculate a weighted average area per unit for the project
        # This is an iterative approach to find total units
        
        # Start with an estimate of total units based on average density
        estimated_total_units = round(predicted_distributable_area * avg_units_per_dist_area)
        
        # Distribute these units based on calculated proportions
        total_area_consumed = 0
        for i, h_type in enumerate(remaining_house_types):
            num_units = round(estimated_total_units * current_proportions[i])
            predicted_units[h_type] = num_units
            
            # Add to total area consumed based on unit type
            if h_type == 'ทาวโฮม':
                total_area_consumed += num_units * AREA_TH
            elif h_type == 'บ้านแฝด':
                total_area_consumed += num_units * AREA_BA
            elif h_type == 'บ้านเดี่ยว':
                total_area_consumed += num_units * AREA_BD
            # Note: For บ้านเดี่ยว3ชั้น and อาคารพาณิชย์, we don't have specified area, so we assume they take up space but don't explicitly calculate their area here.
            # If they exist in grade_rules, they are set to 0. Otherwise, their proportion is used from historical data.

        # Refine total units based on total distributable area and consumed area
        # This is a simple refinement. A more complex model might iterate.
        if total_area_consumed > 0:
            # Re-estimate total units if area_consumed is significantly different
            # We assume a fixed average area for บ้านเดี่ยว3ชั้น and อาคารพาณิชย์ if they are present and not set to 0 by rules.
            # For simplicity, we can use the overall average unit area from historical data if needed.
            
            # Calculate an average unit area for this specific project based on predicted house types
            weighted_avg_unit_area = 0
            if predicted_units['ทาวโฮม'] > 0: weighted_avg_unit_area += predicted_units['ทาวโฮม'] * AREA_TH
            if predicted_units['บ้านแฝด'] > 0: weighted_avg_unit_area += predicted_units['บ้านแฝด'] * AREA_BA
            if predicted_units['บ้านเดี่ยว'] > 0: weighted_avg_unit_area += predicted_units['บ้านเดี่ยว'] * AREA_BD
            
            total_current_estimated_units = sum(predicted_units.values())

            if total_current_estimated_units > 0 and predicted_distributable_area > 0 and weighted_avg_unit_area > 0:
                # Calculate what proportion of total_area_consumed came from our known unit types
                known_types_total_units = predicted_units['ทาวโฮม'] + predicted_units['บ้านแฝด'] + predicted_units['บ้านเดี่ยว']
                
                if known_types_total_units > 0:
                    avg_area_known_types = total_area_consumed / known_types_total_units
                    # Scale based on distributable area for known types
                    estimated_known_units_from_area = predicted_distributable_area / avg_area_known_types
                    scale_factor_known = estimated_known_units_from_area / known_types_total_units if known_types_total_units > 0 else 1.0

                    # Apply scale factor to known types
                    predicted_units['ทาวโฮม'] = round(predicted_units['ทาวโฮม'] * scale_factor_known)
                    predicted_units['บ้านแฝด'] = round(predicted_units['บ้านแฝด'] * scale_factor_known)
                    predicted_units['บ้านเดี่ยว'] = round(predicted_units['บ้านเดี่ยว'] * scale_factor_known)
                    
                    # For other types, we just use the original proportion * new total if they exist
                    # (This part is a bit tricky without their area data, assuming they are minor or fixed by rules)
                    # If they are not 0 by rules and their proportions are available,
                    # we can try to estimate their units based on the remaining area.
                    
                    # For simplicity, let's just sum up the known types for now.
                    # A more robust model would need to account for area of บ้านเดี่ยว3ชั้น and อาคารพาณิชย์
                    # or assume a fixed small footprint for them if they are allowed.
                else: # No known types, distribute remaining based on estimated total
                    for i, h_type in enumerate(remaining_house_types):
                        predicted_units[h_type] = round(estimated_total_units * current_proportions[i])


        total_predicted_units = sum(predicted_units.values())

    else: # No specific grade rules, use average proportions for the given grade/land shape
        if proportions_df is not None:
            # Use specific proportions if available
            for h_type in house_types:
                predicted_units[h_type] = round(predicted_distributable_area * avg_units_per_dist_area * proportions_df[f'{h_type}_prop'].iloc[0])
        else:
            # Fallback to general average proportions if no specific (grade, land_shape) combo found
            general_avg_props = df[[f'{h_type}_prop' for h_type in house_types]].mean()
            for h_type in house_types:
                predicted_units[h_type] = round(predicted_distributable_area * avg_units_per_dist_area * general_avg_props[f'{h_type}_prop'])

        total_predicted_units = sum(predicted_units.values())
        
        # A more robust way to calculate total units given specific area per type
        # Calculate a weighted average area per unit for this project based on predicted proportions
        if total_predicted_units > 0:
            # Recalculate based on proportions and known areas
            # This is complex as it requires an iterative approach or solving for total units
            # Given the constraints, a simpler approach is to use the average density (units/sqm)
            # from historical data and then distribute based on proportions.
            
            # Let's refine based on the area consumption if possible
            estimated_total_area_consumed_by_units = 0
            if predicted_units['ทาวโฮม'] > 0: estimated_total_area_consumed_by_units += predicted_units['ทาวโฮม'] * AREA_TH
            if predicted_units['บ้านแฝด'] > 0: estimated_total_area_consumed_by_units += predicted_units['บ้านแฝด'] * AREA_BA
            if predicted_units['บ้านเดี่ยว'] > 0: estimated_total_area_consumed_by_units += predicted_units['บ้านเดี่ยว'] * AREA_BD

            # If the calculated consumed area is much different from distributable, scale units
            if estimated_total_area_consumed_by_units > 0:
                scale_factor_area = predicted_distributable_area / estimated_total_area_consumed_by_units
                for h_type in ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว']: # Apply only to types with known area
                    predicted_units[h_type] = round(predicted_units[h_type] * scale_factor_area)
                total_predicted_units = sum(predicted_units.values()) # Recalculate total units


    # 3. Predict Number of Alleys (จำนวนซอย)
    predicted_alleys = 0
    if total_predicted_units > 0:
        predicted_alleys = round(total_predicted_units * avg_alley_per_unit)
    elif predicted_distributable_area > 0:
        predicted_alleys = round(predicted_distributable_area * avg_alley_per_dist_area)
    
    # Ensure alleys are at least 1 for any project, maybe more if very large
    predicted_alleys = max(1, predicted_alleys) # At least one alley for a project. Could refine this rule.


    return {
        'พื้นที่โครงการ': round(project_area_sqm, 2),
        'พื้นที่สาธา': round(predicted_public_area, 2),
        'พื้นที่จัดจำหน่าย': round(predicted_distributable_area, 2),
        'พื้นที่สวน': round(predicted_garden_area, 2),
        'พื้นที่ถนนรวม': round(predicted_road_area, 2),
        'จำนวนแปลง (ทาวน์โฮม)': predicted_units['ทาวโฮม'],
        'จำนวนแปลง (บ้านแฝด)': predicted_units['บ้านแฝด'],
        'จำนวนแปลง (บ้านเดี่ยว)': predicted_units['บ้านเดี่ยว'],
        'จำนวนแปลง (บ้านเดี่ยว3ชั้น)': predicted_units['บ้านเดี่ยว3ชั้น'],
        'จำนวนแปลง (อาคารพาณิชย์)': predicted_units['อาคารพาณิชย์'],
        'จำนวนแปลง (รวม)': total_predicted_units,
        'จำนวนซอย': predicted_alleys
    }

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Streamlit UI: File Uploader ---
st.title("📐 Smart Layout Predictor (ML Powered)")
st.write("โปรดอัปโหลดไฟล์ข้อมูลโครงการของคุณ (layoutdata.xlsx - Sheet1.csv) เพื่อเริ่มต้นการทำนาย")

uploaded_file = st.file_uploader("เลือกไฟล์ CSV หรือ Excel (Sheet1.csv)", type=["csv", "xlsx"])

df = None # Initialize df to None

if uploaded_file is not None:
    try:
        # Check file type to read correctly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            # If it's an Excel file, ensure it reads Sheet1
            df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
        else:
            st.error("นามสกุลไฟล์ไม่ถูกต้อง โปรดอัปโหลดไฟล์ .csv หรือ .xlsx")
            st.stop()

        df.columns = df.columns.str.strip() # Remove leading/trailing whitespaces from column names
        st.success("โหลดข้อมูลสำเร็จแล้ว!")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        st.stop()
else:
    st.info("กรุณาอัปโหลดไฟล์ข้อมูลก่อนดำเนินการต่อ")
    st.stop() # Stop execution if no file is uploaded yet

# --- Continue only if df is loaded ---
if df is not None:
    # --- 2. Unit Conversion (sqm ➝ sq. wah) ---
    sqm_to_sqwah = 0.25
    columns_to_convert = ['พื้นที่โครงการ(ตรม)', 'พื้นที่จัดจำหน่าย(ตรม)', 'พื้นที่สาธา(ตรม)', 'พื้นที่ถนนรวม', 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)']
    for col in columns_to_convert:
        if col in df.columns:
            df[col] *= sqm_to_sqwah
        else:
            # ใช้ st.sidebar.warning เพื่อไม่ให้ข้อความไปรบกวนหน้าหลักมากเกินไป
            st.sidebar.warning(f"Column '{col}' not found for unit conversion. Skipping.")

    # --- 3. Setup House Area (ไม่ใช้ในการทำนายโดยตรง แต่เก็บไว้หากต้องการใช้ในอนาคต) ---
    AREA_TH = 5 * 16 * sqm_to_sqwah
    AREA_BA = 10 * 16 * sqm_to_sqwah
    AREA_BD = 15 * 18 * sqm_to_sqwah

    house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']

    # --- 4. Data Preprocessing for Machine Learning ---

    # กำหนด Features (ตัวแปรต้น)
    features = ['พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด']

    # กำหนด Targets (ตัวแปรตาม)
    targets = ['พื้นที่จัดจำหน่าย(ตรม)', 'จำนวนหลัง', 'จำนวนซอย', 'พื้นที่ถนนรวม']
    for h_type in house_types:
        if h_type in df.columns:
            targets.append(h_type)
        else:
            st.sidebar.warning(f"Target column '{h_type}' not found in data. Skipping this target.")

    # Filter out features/targets that are not in the dataframe after loading
    features = [f for f in features if f in df.columns]
    targets = [t for t in targets if t in df.columns]

    if not features:
        st.error("ไม่พบ Features ที่ใช้ในการเทรนโมเดล โปรดตรวจสอบชื่อคอลัมน์ในไฟล์ข้อมูลที่อัปโหลด")
        st.stop()
    if not targets:
        st.error("ไม่พบ Targets ที่ใช้ในการเทรนโมเดล โปรดตรวจสอบชื่อคอลัมน์ในไฟล์ข้อมูลที่อัปโหลด")
        st.stop()

    X = df[features]
    y = df[targets]

    # Handle potential missing categorical values in input data (fill with mode for training)
    for col in ['รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด']:
        if col in X.columns:
            if X[col].isnull().any():
                mode_val = X[col].mode()[0]
                X[col] = X[col].fillna(mode_val)
                st.sidebar.info(f"Filled missing values in '{col}' with mode: {mode_val}")


    categorical_features = [col for col in ['รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด'] if col in X.columns]
    numerical_features = [col for col in ['พื้นที่โครงการ(ตรม)'] if col in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 5. Model Training ---
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
                           ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.sidebar.subheader("สถานะโมเดล")
    with st.spinner('กำลังฝึกโมเดล Machine Learning...'):
        try:
            model.fit(X_train, y_train)
            st.sidebar.success("ฝึกโมเดลเสร็จสิ้น!")
        except Exception as e:
            st.sidebar.error(f"เกิดข้อผิดพลาดในการฝึกโมเดล: {e}")
            st.stop()

    # --- 6. Model Evaluation ---
    st.sidebar.subheader("📊 ประสิทธิภาพโมเดล (บนข้อมูลทดสอบ)")
    try:
        y_pred_test = model.predict(X_test)
        y_pred_test_df = pd.DataFrame(y_pred_test, columns=targets, index=y_test.index)

        for i, target_col in enumerate(targets):
            actual_test_target = y_test[target_col].values
            predicted_test_target = y_pred_test_df[target_col].values

            mae_val = mean_absolute_error(actual_test_target, predicted_test_target)
            r2_val = r2_score(actual_test_target, predicted_test_target)

            st.sidebar.write(f"**Target: {target_col}**")
            st.sidebar.markdown(f"- **MAE:** {mae_val:,.2f}")
            st.sidebar.markdown(f"- **R²:** {r2_val:.4f}")
    except Exception as e:
        st.sidebar.error(f"เกิดข้อผิดพลาดในการประเมินโมเดล: {e}")


    # --- 7. Prediction Function (ใช้โมเดล ML) ---
    def predict_with_ml(project_area, land_shape, grade, province, ml_model, house_types_list, target_cols, original_df):
        input_data = pd.DataFrame([[project_area, land_shape, grade, province]],
                                  columns=['พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด'])

        predicted_values = ml_model.predict(input_data)[0]

        predicted_dict = {target: value for target, value in zip(target_cols, predicted_values)}

        result = {
            'พื้นที่โครงการ (ตร.วา)': round(project_area, 2),
            'พื้นที่ขาย (ตร.วา)': round(predicted_dict.get('พื้นที่จัดจำหน่าย(ตรม)', 0), 2),
            'พื้นที่ถนน (ตร.วา)': round(predicted_dict.get('พื้นที่ถนนรวม', 0), 2),
            'จำนวนแปลงรวม': max(0, int(round(predicted_dict.get('จำนวนหลัง', 0)))),
            'จำนวนซอย': max(1, int(round(predicted_dict.get('จำนวนซอย', 0))))
        }

        for h_type in house_types_list:
            col_name = h_type
            if col_name in predicted_dict:
                result[f'จำนวนแปลง ({h_type})'] = max(0, int(round(predicted_dict[col_name])))
            else:
                result[f'จำนวนแปลง ({h_type})'] = 0

        # คำนวณพื้นที่อื่นๆ (พื้นที่สาธารณะ, พื้นที่สวน)
        # หาก 'พื้นที่สาธา(ตรม)' และ 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)' ไม่ได้อยู่ใน targets
        # เราจะใช้ค่าเฉลี่ยสัดส่วนจากข้อมูลเดิม
        if 'พื้นที่โครงการ(ตรม)' in original_df.columns and 'พื้นที่จัดจำหน่าย(ตรม)' in original_df.columns:
            # ลองใช้ค่าที่ทำนายได้จาก model ถ้ามี target นี้
            public_area_predicted = predicted_dict.get('พื้นที่สาธา(ตรม)', None)
            if public_area_predicted is not None:
                 result['พื้นที่สาธารณะ (ตร.วา)'] = round(public_area_predicted, 2)
            else:
                # Fallback to historical average if not predicted
                avg_public_area_ratio = original_df['พื้นที่สาธา(ตรม)'].sum() / original_df['พื้นที่โครงการ(ตรม)'].sum() if original_df['พื้นที่โครงการ(ตรม)'].sum() > 0 else 0
                result['พื้นที่สาธารณะ (ตร.วา)'] = round(project_area * avg_public_area_ratio, 2)

            avg_garden_ratio_of_dist = 0.05 # ค่าคงที่ 5%
            result['พื้นที่สวน (ตร.วา)'] = round(result['พื้นที่ขาย (ตร.วา)'] * avg_garden_ratio_of_dist, 2)
        else:
            st.sidebar.warning("ไม่สามารถคำนวณพื้นที่สาธารณะและพื้นที่สวนได้เนื่องจากคอลัมน์ต้นฉบับขาดหายไป")
            result['พื้นที่สาธารณะ (ตร.วา)'] = 0
            result['พื้นที่สวน (ตร.วา)'] = 0

        return result

    # --- 8. Streamlit UI: Input and Prediction ---
    st.write("ป้อนข้อมูลโครงการเพื่อทำนายผังและจำนวนแปลงโดยใช้โมเดล Machine Learning.")

    # Ensure unique values for selectboxes are taken from the loaded data's columns
    project_area_input = st.number_input("พื้นที่โครงการ (ตร.วา)", min_value=float(X['พื้นที่โครงการ(ตรม)'].min()) if 'พื้นที่โครงการ(ตรม)' in X.columns else 1000.0,
                                         max_value=float(X['พื้นที่โครงการ(ตรม)'].max()) if 'พื้นที่โครงการ(ตรม)' in X.columns else 100000.0,
                                         value=float(X['พื้นที่โครงการ(ตรม)'].mean()) if 'พื้นที่โครงการ(ตรม)' in X.columns else 40000.0, step=500.0)

    # Use .dropna().unique() to get valid options for selectbox, excluding NaN
    land_shape_options = X['รูปร่างที่ดิน'].dropna().unique().tolist() if 'รูปร่างที่ดิน' in X.columns else ['ไม่ระบุ']
    if 'ไม่ระบุ' not in land_shape_options and 'ไม่ระบุ' in df['รูปร่างที่ดิน'].unique(): # If 'ไม่ระบุ' existed in original data
        land_shape_options.insert(0, 'ไม่ระบุ') # Add it to the start
    land_shape_input = st.selectbox("รูปร่างที่ดิน", land_shape_options)

    grade_options = X['เกรดโครงการ'].dropna().unique().tolist() if 'เกรดโครงการ' in X.columns else ['ไม่ระบุ']
    if 'ไม่ระบุ' not in grade_options and 'ไม่ระบุ' in df['เกรดโครงการ'].unique():
        grade_options.insert(0, 'ไม่ระบุ')
    grade_input = st.selectbox("เกรดโครงการ", grade_options)

    province_options = X['จังหวัด'].dropna().unique().tolist() if 'จังหวัด' in X.columns else ['ไม่ระบุ']
    if 'ไม่ระบุ' not in province_options and 'ไม่ระบุ' in df['จังหวัด'].unique():
        province_options.insert(0, 'ไม่ระบุ')
    province_input = st.selectbox("จังหวัด", province_options)

    if st.button("ทำนายผังโครงการด้วย ML"):
        # Basic validation for selected options
        if land_shape_input not in land_shape_options or \
           grade_input not in grade_options or \
           province_input not in province_options:
            st.warning("ค่าที่คุณเลือกบางค่าอาจไม่อยู่ในข้อมูลที่ใช้ฝึกโมเดล ผลลัพธ์อาจไม่แม่นยำ")

        result_ml = predict_with_ml(
            project_area_input,
            land_shape_input,
            grade_input,
            province_input,
            model,
            house_types,
            targets,
            df # Pass original df to calculate non-predicted areas
        )
        st.subheader("🔍 ผลการทำนายจาก ML")
        st.dataframe(pd.DataFrame(result_ml.items(), columns=['รายการ', 'ค่าทำนาย']), use_container_width=True)

        st.info("โปรดทราบ: ค่า MAE และ R² ที่แสดงในแถบด้านข้างคือประสิทธิภาพของโมเดลบนข้อมูลทดสอบ ไม่ใช่ผลลัพธ์จากการทำนายเดี่ยวๆ นี้")

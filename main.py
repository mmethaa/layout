import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV # Added GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb # Import XGBoost

# --- Streamlit UI: File Uploader ---
st.title("📐 Smart Layout Predictor (ML Powered)")
st.write("โปรดอัปโหลดไฟล์ข้อมูลโครงการของคุณ (layoutdata.xlsx - Sheet1.csv) เพื่อเริ่มต้นการทำนาย")

uploaded_file = st.file_uploader("เลือกไฟล์ CSV หรือ Excel (Sheet1.csv)", type=["csv", "xlsx"])

df = None # Initialize df to None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
        else:
            st.error("นามสกุลไฟล์ไม่ถูกต้อง โปรดอัปโหลดไฟล์ .csv หรือ .xlsx")
            st.stop()

        df.columns = df.columns.str.strip()
        st.success("โหลดข้อมูลสำเร็จแล้ว!")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        st.stop()
else:
    st.info("กรุณาอัปโหลดไฟล์ข้อมูลก่อนดำเนินการต่อ")
    st.stop()

# --- Continue only if df is loaded ---
if df is not None:
    # --- 2. Unit Conversion (sqm ➝ sq. wah) ---
    sqm_to_sqwah = 0.25
    columns_to_convert = ['พื้นที่โครงการ(ตรม)', 'พื้นที่จัดจำหน่าย(ตรม)', 'พื้นที่สาธา(ตรม)', 'พื้นที่ถนนรวม', 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)']
    for col in columns_to_convert:
        if col in df.columns:
            df[col] *= sqm_to_sqwah
        else:
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
    # เพิ่ม 'พื้นที่สาธา(ตรม)' เพื่อให้โมเดลทำนายพื้นที่สาธารณะโดยตรง
    targets = ['พื้นที่จัดจำหน่าย(ตรม)', 'จำนวนหลัง', 'จำนวนซอย', 'พื้นที่ถนนรวม', 'พื้นที่สาธา(ตรม)']
    for h_type in house_types:
        if h_type in df.columns:
            targets.append(h_type)
        else:
            st.sidebar.warning(f"Target column '{h_type}' not found in data. Skipping this target.")


    features = [f for f in features if f in df.columns]
    targets = [t for t in targets if t in t in df.columns]

    if not features:
        st.error("ไม่พบ Features ที่ใช้ในการเทรนโมเดล โปรดตรวจสอบชื่อคอลัมน์ในไฟล์ข้อมูลที่อัปโหลด")
        st.stop()
    if not targets:
        st.error("ไม่พบ Targets ที่ใช้ในการเทรนโมเดล โปรดตรวจสอบชื่อคอลัมน์ในไฟล์ข้อมูลที่อัปโหลด")
        st.stop()

    X = df[features]
    y = df[targets]

    # Handle potential missing categorical values in input data (fill with mode for training)
    # Ensure all features have no NaN before processing with pipeline
    for col in ['รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด']:
        if col in X.columns:
            if X[col].isnull().any():
                mode_val = X[col].mode()[0]
                X[col] = X[col].fillna(mode_val)
                st.sidebar.info(f"Filled missing values in '{col}' with mode: {mode_val}")

    # Handle potential missing numerical values (e.g., in พื้นที่โครงการ(ตรม) if any)
    for col in ['พื้นที่โครงการ(ตรม)']:
        if col in X.columns:
            if X[col].isnull().any():
                mean_val = X[col].mean()
                X[col] = X[col].fillna(mean_val)
                st.sidebar.info(f"Filled missing values in '{col}' with mean: {mean_val:.2f}")


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

    # Allow user to select model type
    st.sidebar.subheader("ตัวเลือกโมเดล")
    model_type = st.sidebar.selectbox("เลือกประเภทโมเดล", ["RandomForestRegressor", "XGBRegressor"])

    if model_type == "RandomForestRegressor":
        regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        # Parameters for GridSearchCV (can be tuned)
        # param_grid = {
        #     'regressor__n_estimators': [100, 200, 300],
        #     'regressor__max_depth': [None, 10, 20],
        #     'regressor__min_samples_split': [2, 5],
        # }
    elif model_type == "XGBRegressor":
        # XGBoost handles multi-output directly, but requires a slight wrapper for scikit-learn pipeline for multiple targets
        # For multi-output with XGBoost, it's often better to train separate models for each target
        # Or use a multi-output wrapper if the targets are independent enough for this.
        # For simplicity with pipeline, we'll let XGBoost handle it, which it does well for common cases.
        regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42, n_jobs=-1)
        # Parameters for GridSearchCV (can be tuned)
        # param_grid = {
        #     'regressor__n_estimators': [100, 200, 300],
        #     'regressor__learning_rate': [0.05, 0.1, 0.2],
        #     'regressor__max_depth': [3, 5, 7],
        # }

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', regressor)
                           ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.sidebar.subheader("สถานะโมเดล")
    with st.spinner(f'กำลังฝึกโมเดล {model_type}...'):
        try:
            model.fit(X_train, y_train)
            st.sidebar.success(f"ฝึกโมเดล {model_type} เสร็จสิ้น!")

            # Example of how to use GridSearchCV (uncomment if you want to run it, it takes time)
            # if 'param_grid' in locals(): # Check if param_grid was defined for the selected model
            #     st.sidebar.write("กำลังทำการ Hyperparameter Tuning (อาจใช้เวลา)...")
            #     grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='r2', verbose=1)
            #     grid_search.fit(X_train, y_train)
            #     st.sidebar.success(f"Tuning เสร็จสิ้น! Best R2: {grid_search.best_score_:.4f}")
            #     st.sidebar.write("Best parameters:", grid_search.best_params_)
            #     model = grid_search.best_estimator_ # Use the best model found

        except Exception as e:
            st.sidebar.error(f"เกิดข้อผิดพลาดในการฝึกโมเดล: {e}")
            st.stop()

    # --- 6. Model Evaluation ---
    st.sidebar.subheader("📊 ประสิทธิภาพโมเดล (บนข้อมูลทดสอบ)")
    try:
        y_pred_test = model.predict(X_test)
        # Ensure y_pred_test is a DataFrame for consistent column access
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
    def predict_with_ml(project_area, land_shape, grade, province, ml_model, house_types_list, target_cols):
        input_data = pd.DataFrame([[project_area, land_shape, grade, province]],
                                  columns=['พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด'])

        predicted_values = ml_model.predict(input_data)[0]

        predicted_dict = {target: value for target, value in zip(target_cols, predicted_values)}

        result = {
            'พื้นที่โครงการ (ตร.วา)': round(project_area, 2),
            'พื้นที่ขาย (ตร.วา)': round(predicted_dict.get('พื้นที่จัดจำหน่าย(ตรม)', 0), 2),
            'พื้นที่ถนน (ตร.วา)': round(predicted_dict.get('พื้นที่ถนนรวม', 0), 2),
            'พื้นที่สาธารณะ (ตร.วา)': round(predicted_dict.get('พื้นที่สาธา(ตรม)', 0), 2), # ใช้ค่าที่ทำนายได้โดยตรง
            'จำนวนแปลงรวม': max(0, int(round(predicted_dict.get('จำนวนหลัง', 0)))),
            'จำนวนซอย': max(1, int(round(predicted_dict.get('จำนวนซอย', 0))))
        }

        for h_type in house_types_list:
            col_name = h_type
            if col_name in predicted_dict:
                result[f'จำนวนแปลง ({h_type})'] = max(0, int(round(predicted_dict[col_name])))
            else:
                result[f'จำนวนแปลง ({h_type})'] = 0

        # คำนวณพื้นที่สวน: ยังคงใช้ 5% ของพื้นที่ขายที่ทำนายได้
        avg_garden_ratio_of_dist = 0.05
        result['พื้นที่สวน (ตร.วา)'] = round(result['พื้นที่ขาย (ตร.วา)'] * avg_garden_ratio_of_dist, 2)

        return result

    # --- 8. Streamlit UI: Input and Prediction ---
    st.write("ป้อนข้อมูลโครงการเพื่อทำนายผังและจำนวนแปลงโดยใช้โมเดล Machine Learning.")

    # Input fields
    # Ensure min/max/default values for number_input are floats and handle cases where X might be empty initially
    project_area_input = st.number_input("พื้นที่โครงการ (ตร.วา)",
                                         min_value=float(X['พื้นที่โครงการ(ตรม)'].min()) if 'พื้นที่โครงการ(ตรม)' in X.columns and not X['พื้นที่โครงการ(ตรม)'].empty else 1000.0,
                                         max_value=float(X['พื้นที่โครงการ(ตรม)'].max()) if 'พื้นที่โครงการ(ตรม)' in X.columns and not X['พื้นที่โครงการ(ตรม)'].empty else 100000.0,
                                         value=float(X['พื้นที่โครงการ(ตรม)'].mean()) if 'พื้นที่โครงการ(ตรม)' in X.columns and not X['พื้นที่โครงการ(ตรม)'].empty else 40000.0,
                                         step=500.0)

    land_shape_options = X['รูปร่างที่ดิน'].dropna().unique().tolist() if 'รูปร่างที่ดิน' in X.columns and not X['รูปร่างที่ดิน'].empty else ['ไม่ระบุ']
    land_shape_input = st.selectbox("รูปร่างที่ดิน", land_shape_options)

    grade_options = X['เกรดโครงการ'].dropna().unique().tolist() if 'เกรดโครงการ' in X.columns and not X['เกรดโครงการ'].empty else ['ไม่ระระบุ']
    grade_input = st.selectbox("เกรดโครงการ", grade_options)

    province_options = X['จังหวัด'].dropna().unique().tolist() if 'จังหวัด' in X.columns and not X['จังหวัด'].empty else ['ไม่ระบุ']
    province_input = st.selectbox("จังหวัด", province_options)

    if st.button("ทำนายผังโครงการด้วย ML"):
        # Basic validation for selected options
        # Check if selected values are actually in the options list for safety
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
            targets
        )
        st.subheader("🔍 ผลการทำนายจาก ML")
        st.dataframe(pd.DataFrame(result_ml.items(), columns=['รายการ', 'ค่าทำนาย']), use_container_width=True)

        st.info("โปรดทราบ: ค่า MAE และ R² ที่แสดงในแถบด้านข้างคือประสิทธิภาพของโมเดลบนข้อมูลทดสอบ ไม่ใช่ผลลัพธ์จากการทำนายเดี่ยวๆ นี้")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# --- Streamlit UI: File Uploader ---
st.title("📐 Smart Layout Predictor (ML Powered)")
st.write("โปรดอัปโหลดไฟล์ข้อมูลโครงการของคุณ (layoutdata.xlsx - Sheet1.csv) เพื่อเริ่มต้นการทำนายและวิเคราะห์")

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
            st.sidebar.warning(f"Column '{col}' not found for unit conversion. Skipping.")

    # --- 3. Setup House Area (ไม่ใช้ในการทำนายโดยตรง แต่เก็บไว้หากต้องการใช้ในอนาคต) ---
    AREA_TH = 5 * 16 * sqm_to_sqwah
    AREA_BA = 10 * 16 * sqm_to_sqwah
    AREA_BD = 15 * 18 * sqm_to_sqwah

    house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']

    # --- NEW: สมมติฐานราคาขายและต้นทุนต่อหน่วย (สามารถปรับเปลี่ยนได้ตามข้อมูลจริง) ---
    st.sidebar.subheader("สมมติฐานราคาและต้นทุน")
    st.sidebar.write("ใช้สำหรับการคำนวณรายได้และกำไร")

    # ราคาขายต่อหลัง (เฉลี่ย)
    st.sidebar.markdown("**ราคาขายต่อหลัง (บาท)**")
    sale_price_th = st.sidebar.number_input("ทาวโฮม", value=2_500_000, step=100_000)
    sale_price_ba = st.sidebar.number_input("บ้านแฝด", value=4_000_000, step=100_000)
    sale_price_bd = st.sidebar.number_input("บ้านเดี่ยว", value=6_000_000, step=100_000)
    sale_price_bd3 = st.sidebar.number_input("บ้านเดี่ยว3ชั้น", value=8_000_000, step=100_000)
    sale_price_ap = st.sidebar.number_input("อาคารพาณิชย์", value=5_000_000, step=100_000)

    sale_price_per_unit = {
        'ทาวโฮม': sale_price_th,
        'บ้านแฝด': sale_price_ba,
        'บ้านเดี่ยว': sale_price_bd,
        'บ้านเดี่ยว3ชั้น': sale_price_bd3,
        'อาคารพาณิชย์': sale_price_ap
    }

    # ต้นทุนก่อสร้างต่อหลัง (เฉลี่ย)
    st.sidebar.markdown("**ต้นทุนก่อสร้างต่อหลัง (บาท)**")
    cost_th = st.sidebar.number_input("ทาวโฮม_Cost", value=1_500_000, step=50_000)
    cost_ba = st.sidebar.number_input("บ้านแฝด_Cost", value=2_500_000, step=50_000)
    cost_bd = st.sidebar.number_input("บ้านเดี่ยว_Cost", value=3_500_000, step=50_000)
    cost_bd3 = st.sidebar.number_input("บ้านเดี่ยว3ชั้น_Cost", value=5_000_000, step=50_000)
    cost_ap = st.sidebar.number_input("อาคารพาณิชย์_Cost", value=3_000_000, step=50_000)

    construction_cost_per_unit = {
        'ทาวโฮม': cost_th,
        'บ้านแฝด': cost_ba,
        'บ้านเดี่ยว': cost_bd,
        'บ้านเดี่ยว3ชั้น': cost_bd3,
        'อาคารพาณิชย์': cost_ap
    }

    # ต้นทุนที่ดินและพัฒนาอื่นๆ (ค่าต่อ ตร.วา และ % ของรายได้รวม)
    land_cost_per_sqwah = st.sidebar.number_input("ต้นทุนที่ดิน (บาท/ตร.วา)", value=50000, step=1000)
    other_development_cost_ratio = st.sidebar.slider("ต้นทุนพัฒนาอื่นๆ (% ของรายได้รวม)", min_value=0.0, max_value=0.3, value=0.20, step=0.01)

    # --- 4. Data Preprocessing for Machine Learning ---

    # กำหนด Features (ตัวแปรต้น)
    features = ['พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด']

    # กำหนด Targets (ตัวแปรตาม)
    targets = ['พื้นที่จัดจำหน่าย(ตรม)', 'จำนวนหลัง', 'จำนวนซอย', 'พื้นที่ถนนรวม', 'พื้นที่สาธา(ตรม)']
    for h_type in house_types:
        if h_type in df.columns:
            targets.append(h_type)
        else:
            st.sidebar.warning(f"Target column '{h_type}' not found in data. Skipping this target.")

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

    for col in ['รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด']:
        if col in X.columns:
            if X[col].isnull().any():
                mode_val = X[col].mode()[0]
                X[col] = X[col].fillna(mode_val)
                st.sidebar.info(f"Filled missing values in '{col}' with mode: {mode_val}")

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
    st.sidebar.subheader("ตัวเลือกโมเดล")
    model_type = st.sidebar.selectbox("เลือกประเภทโมเดล", ["RandomForestRegressor", "XGBRegressor"])

    if model_type == "RandomForestRegressor":
        regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    elif model_type == "XGBRegressor":
        regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42, n_jobs=-1)

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', regressor)
                           ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.sidebar.subheader("สถานะโมเดล")
    with st.spinner(f'กำลังฝึกโมเดล {model_type}...'):
        try:
            model.fit(X_train, y_train)
            st.sidebar.success(f"ฝึกโมเดล {model_type} เสร็จสิ้น!")
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


    # --- 7. Prediction Function (ใช้โมเดล ML และคำนวณรายได้/กำไร) ---
    def predict_with_ml(project_area, land_shape, grade, province, ml_model, house_types_list, target_cols,
                        sale_prices, construct_costs, land_cost_sqwah, other_dev_ratio):
        input_data = pd.DataFrame([[project_area, land_shape, grade, province]],
                                  columns=['พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด'])

        predicted_values = ml_model.predict(input_data)[0]

        predicted_dict = {target: value for target, value in zip(target_cols, predicted_values)}

        result = {
            'พื้นที่โครงการ (ตร.วา)': round(project_area, 2),
            'พื้นที่ขาย (ตร.วา)': round(predicted_dict.get('พื้นที่จัดจำหน่าย(ตรม)', 0), 2),
            'พื้นที่ถนน (ตร.วา)': round(predicted_dict.get('พื้นที่ถนนรวม', 0), 2),
            'พื้นที่สาธารณะ (ตร.วา)': round(predicted_dict.get('พื้นที่สาธา(ตรม)', 0), 2),
            'จำนวนแปลงรวม': max(0, int(round(predicted_dict.get('จำนวนหลัง', 0)))),
            'จำนวนซอย': max(1, int(round(predicted_dict.get('จำนวนซอย', 0))))
        }

        total_gross_revenue = 0
        total_construction_cost = 0

        for h_type in house_types_list:
            col_name = h_type
            num_units = max(0, int(round(predicted_dict.get(col_name, 0))))
            result[f'จำนวนแปลง ({h_type})'] = num_units

            # คำนวณรายได้และต้นทุนก่อสร้างจากจำนวนแปลงที่ทำนายได้
            total_gross_revenue += num_units * sale_prices.get(h_type, 0)
            total_construction_cost += num_units * construct_costs.get(h_type, 0)
        
        # คำนวณต้นทุนที่ดิน
        calculated_land_cost = project_area * land_cost_sqwah # project_area คือ ตร.วา แล้ว

        # คำนวณต้นทุนพัฒนาอื่นๆ (คิดเป็น % ของรายได้รวม)
        calculated_other_dev_cost = total_gross_revenue * other_dev_ratio

        total_cost = total_construction_cost + calculated_land_cost + calculated_other_dev_cost
        profit = total_gross_revenue - total_cost

        result['รายได้รวม (ทำนาย)'] = round(total_gross_revenue, 2)
        result['ต้นทุนรวม (ทำนาย)'] = round(total_cost, 2)
        result['กำไร (ทำนาย)'] = round(profit, 2)

        avg_garden_ratio_of_dist = 0.05
        result['พื้นที่สวน (ตร.วา)'] = round(result['พื้นที่ขาย (ตร.วา)'] * avg_garden_ratio_of_dist, 2)

        return result

    # --- 8. Best Option Analysis ---
    st.subheader("💡 วิเคราะห์ทางเลือกที่ดีที่สุด (อิงจากข้อมูลในอดีตและสมมติฐานกำไร)")
    st.write("วิเคราะห์เกรดโครงการที่ให้ 'กำไร (ทำนาย)' สูงสุด เมื่อพิจารณาจากรูปร่างที่ดินและจังหวัด")

    available_grades_for_analysis = df['เกรดโครงการ'].dropna().unique().tolist()
    if not available_grades_for_analysis:
        st.warning("ไม่พบข้อมูล 'เกรดโครงการ' ในไฟล์ที่อัปโหลด ไม่สามารถวิเคราะห์ได้")
    else:
        analysis_land_shape = st.selectbox("รูปร่างที่ดินสำหรับวิเคราะห์", df['รูปร่างที่ดิน'].dropna().unique().tolist(), key='analysis_land_shape')
        analysis_province = st.selectbox("จังหวัดสำหรับวิเคราะห์", df['จังหวัด'].dropna().unique().tolist(), key='analysis_province')
        analysis_project_area = st.number_input("พื้นที่โครงการ (ตร.วา) สำหรับวิเคราะห์", min_value=1000.0, max_value=100000.0, value=40000.0, step=500.0, key='analysis_project_area')

        if st.button("ค้นหาทางเลือกที่ดีที่สุด (กำไรสูงสุด)"):
            best_grade_results = []
            for grade_option in available_grades_for_analysis:
                predicted_output = predict_with_ml(
                    analysis_project_area,
                    analysis_land_shape,
                    grade_option,
                    analysis_province,
                    model,
                    house_types,
                    targets,
                    sale_price_per_unit,       # Pass assumptions
                    construction_cost_per_unit,# Pass assumptions
                    land_cost_per_sqwah,       # Pass assumptions
                    other_development_cost_ratio # Pass assumptions
                )
                best_grade_results.append({
                    'เกรดโครงการ': grade_option,
                    'จำนวนแปลงรวม (ทำนาย)': predicted_output['จำนวนแปลงรวม'],
                    'พื้นที่ขาย (ตร.วา) (ทำนาย)': predicted_output['พื้นที่ขาย (ตร.วา)'],
                    'รายได้รวม (ทำนาย)': predicted_output['รายได้รวม (ทำนาย)'],
                    'ต้นทุนรวม (ทำนาย)': predicted_output['ต้นทุนรวม (ทำนาย)'],
                    'กำไร (ทำนาย)': predicted_output['กำไร (ทำนาย)'],
                    # Add other details if necessary for comparison
                    'จำนวนแปลง (ทาวโฮม) (ทำนาย)': predicted_output['จำนวนแปลง (ทาวโฮม)'],
                    'จำนวนแปลง (บ้านแฝด) (ทำนาย)': predicted_output['จำนวนแปลง (บ้านแฝด)'],
                    'จำนวนแปลง (บ้านเดี่ยว) (ทำนาย)': predicted_output['จำนวนแปลง (บ้านเดี่ยว)'],
                })

            results_df = pd.DataFrame(best_grade_results)
            # Sort by 'กำไร (ทำนาย)' to find the best option for profit
            results_df_sorted = results_df.sort_values(by='กำไร (ทำนาย)', ascending=False).reset_index(drop=True)

            st.write("### ผลการทำนายและจัดอันดับตาม 'กำไร (ทำนาย)'")
            st.dataframe(results_df_sorted, use_container_width=True)

            if not results_df_sorted.empty:
                best_option = results_df_sorted.iloc[0]
                st.success(f"**ทางเลือกที่ดีที่สุดคือ: เกรดโครงการ '{best_option['เกรดโครงการ']}'** "
                           f"ซึ่งคาดว่าจะให้ 'กำไร' สูงสุดที่: **{best_option['กำไร (ทำนาย)']}:,.2f บาท**")
            else:
                st.warning("ไม่สามารถระบุทางเลือกที่ดีที่สุดได้จากข้อมูลที่ทำนาย")

    # --- 9. Streamlit UI: Input and Prediction (for single project prediction) ---
    st.subheader("📈 ทำนายผังโครงการสำหรับโครงการใหม่")
    st.write("ป้อนข้อมูลโครงการเพื่อทำนายผัง, จำนวนแปลง, รายได้, และกำไร โดยใช้โมเดล Machine Learning.")

    project_area_input = st.number_input("พื้นที่โครงการ (ตร.วา)",
                                         min_value=float(X['พื้นที่โครงการ(ตรม)'].min()) if 'พื้นที่โครงการ(ตรม)' in X.columns and not X['พื้นที่โครงการ(ตรม)'].empty else 1000.0,
                                         max_value=float(X['พื้นที่โครงการ(ตรม)'].max()) if 'พื้นที่โครงการ(ตรม)' in X.columns and not X['พื้นที่โครงการ(ตรม)'].empty else 100000.0,
                                         value=float(X['พื้นที่โครงการ(ตรม)'].mean()) if 'พื้นที่โครงการ(ตรม)' in X.columns and not X['พื้นที่โครงการ(ตรม)'].empty else 40000.0,
                                         step=500.0, key='single_predict_project_area')

    land_shape_options = X['รูปร่างที่ดิน'].dropna().unique().tolist() if 'รูปร่างที่ดิน' in X.columns and not X['รูปร่างที่ดิน'].empty else ['ไม่ระบุ']
    land_shape_input = st.selectbox("รูปร่างที่ดิน", land_shape_options, key='single_predict_land_shape')

    grade_options = X['เกรดโครงการ'].dropna().unique().tolist() if 'เกรดโครงการ' in X.columns and not X['เกรดโครงการ'].empty else ['ไม่ระบุ']
    grade_input = st.selectbox("เกรดโครงการ", grade_options, key='single_predict_grade')

    province_options = X['จังหวัด'].dropna().unique().tolist() if 'จังหวัด' in X.columns and not X['จังหวัด'].empty else ['ไม่ระบุ']
    province_input = st.selectbox("จังหวัด", province_options, key='single_predict_province')

    if st.button("ทำนายผังโครงการเฉพาะ"):
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
            sale_price_per_unit,       # Pass assumptions
            construction_cost_per_unit,# Pass assumptions
            land_cost_per_sqwah,       # Pass assumptions
            other_development_cost_ratio # Pass assumptions
        )
        st.subheader("🔍 ผลการทำนายจาก ML (โครงการเฉพาะ)")
        st.dataframe(pd.DataFrame(result_ml.items(), columns=['รายการ', 'ค่าทำนาย']), use_container_width=True)

        st.info("โปรดทราบ: ค่า MAE และ R² ที่แสดงในแถบด้านข้างคือประสิทธิภาพของโมเดลบนข้อมูลทดสอบ ไม่ใช่ผลลัพธ์จากการทำนายเดี่ยวๆ นี้")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import requests
import json
from matplotlib import font_manager as fm

# --- API Configuration ---
GEMINI_API_KEY = ""
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Constants and File Paths ---
MODEL_PATH = "random_forest_model.joblib"
THAI_FONT_FILE = "Sarabun-Regular.ttf"

# --- Helper Functions ---
@st.cache_resource
def call_gemini_api(prompt):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    api_key_to_use = GEMINI_API_KEY
    if not api_key_to_use:
        pass
    url = f"{GEMINI_API_URL}?key={api_key_to_use}"
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_json = response.json()
        if 'candidates' in response_json and response_json['candidates']:
            return response_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return "ไม่สามารถสร้างสรุปได้ โปรดลองอีกครั้ง"
    except requests.exceptions.RequestException as e:
        st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ Gemini API: {e}")
        return "ไม่สามารถสร้างสรุปได้ เนื่องจากเกิดข้อผิดพลาดในการเชื่อมต่อ"
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
        return "ไม่สามารถสร้างสรุปได้ เนื่องจากเกิดข้อผิดพลาดที่ไม่คาดคิด"

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
        df.columns = df.columns.str.strip()
        st.success("โหลดข้อมูลสำเร็จแล้ว!")
        return df
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        return None

def plot_area_pie_chart(data, labels):
    fig, ax = plt.subplots()
    ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#1d4ed8', '#60a5fa', '#34d399', '#fbbf24'])
    ax.axis('equal')
    st.pyplot(fig)

def plot_house_bar_chart(result_ml, house_types_list):
    house_data = []
    house_labels = []
    for h_type in house_types_list:
        count = result_ml.get(f'จำนวนแปลง ({h_type})', 0)
        if count > 0:
            house_data.append(count)
            house_labels.append(h_type)
    if house_data:
        fig, ax = plt.subplots()
        ax.bar(house_labels, house_data, color=['#1d4ed8', '#3b82f6', '#60a5fa', '#93c5fd', '#f97316'])
        ax.set_ylabel('จำนวนแปลง')
        ax.set_title('จำนวนแปลงบ้านแต่ละประเภท')
        st.pyplot(fig)
    else:
        st.warning("ไม่มีข้อมูลจำนวนแปลงบ้านที่ทำนายได้")

def get_historically_present_house_types(df, grade_input):
    grade_df = df[df['เกรดโครงการ'] == grade_input]
    present_house_types = set()
    house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']
    for h_type in house_types:
        if h_type in grade_df.columns and grade_df[h_type].sum() > 0:
            present_house_types.add(h_type)
    return present_house_types

@st.cache_data(show_spinner=False)
def predict_and_analyze(project_area, land_shape, grade, province, ml_model, house_types_list, target_cols,
                        sale_prices, construct_costs, land_cost_sqwah, other_dev_ratio,
                        unit_standard_area_sqwah_dict, historically_present_types):
    """
    Performs a prediction using the trained ML model, applies a correction for house types,
    and calculates financial metrics.
    """
    input_data = pd.DataFrame([[project_area, land_shape, grade, province]],
                              columns=['พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด'])
    
    predicted_values = ml_model.predict(input_data)[0]
    predicted_dict = {target: target_value for target, target_value in zip(target_cols, predicted_values)}

    for h_type in house_types_list:
        if h_type not in historically_present_types:
            if h_type in predicted_dict:
                predicted_dict[h_type] = 0

    result = {
        'พื้นที่โครงการ (ตร.วา)': round(project_area, 2),
        'พื้นที่ขาย (ตร.วา)': round(predicted_dict.get('พื้นที่จัดจำหน่าย(ตรม)', 0), 2),
        'พื้นที่ถนน (ตร.วา)': round(predicted_dict.get('พื้นที่ถนนรวม', 0), 2),
        'พื้นที่สาธารณะ (ตร.วา)': round(predicted_dict.get('พื้นที่สาธา(ตรม)', 0), 2),
        'จำนวนแปลงรวม': max(0, int(round(predicted_dict.get('จำนวนหลัง', 0)))),
        'จำนวนซอย': max(1, int(round(predicted_dict.get('จำนวนซอย', 0))))
    }
    
    avg_garden_ratio_of_dist = 0.05
    result['พื้นที่สวน (ตร.วา)'] = round(result['พื้นที่ขาย (ตร.วา)'] * avg_garden_ratio_of_dist, 2)
    
    total_gross_revenue = 0
    total_construction_cost = 0
    total_area_from_units = 0

    for h_type in house_types_list:
        col_name = h_type
        num_units = max(0, int(round(predicted_dict.get(col_name, 0))))
        result[f'จำนวนแปลง ({h_type})'] = num_units
        total_gross_revenue += num_units * sale_prices.get(h_type, 0)
        total_construction_cost += num_units * construct_costs.get(h_type, 0)
        if h_type in unit_standard_area_sqwah_dict:
            total_area_from_units += num_units * unit_standard_area_sqwah_dict[h_type]
    
    calculated_land_cost = project_area * land_cost_sqwah
    calculated_other_dev_cost = total_gross_revenue * other_dev_ratio
    total_cost = total_construction_cost + calculated_land_cost + calculated_other_dev_cost
    profit = total_gross_revenue - total_cost
    
    result['รายได้รวม (ทำนาย)'] = round(total_gross_revenue, 2)
    result['ต้นทุนรวม (ทำนาย)'] = round(total_cost, 2)
    result['กำไร (ทำนาย)'] = round(profit, 2)
    result['พื้นที่รวมจากจำนวนแปลง (ตร.วา)'] = round(total_area_from_units, 2)
    
    predicted_sale_area = result['พื้นที่ขาย (ตร.วา)']
    if predicted_sale_area > 0:
        area_diff = predicted_sale_area - total_area_from_units
        result['ส่วนต่าง (พื้นที่ขาย - พื้นที่รวมแปลง)'] = round(area_diff, 2)
        utilization_ratio = total_area_from_units / predicted_sale_area
        result['สัดส่วนพื้นที่แปลงที่ใช้จากพื้นที่ขาย (%)'] = round(utilization_ratio * 100, 2)
        if utilization_ratio > 1.05:
            st.warning(f"💡 คำเตือน: พื้นที่รวมจากจำนวนแปลงที่ทำนาย ({total_area_from_units:,.2f} ตร.วา) "
                        f"สูงกว่าพื้นที่ขายที่โมเดลทำนายไว้ ({predicted_sale_area:,.2f} ตร.วา) "
                        f"อย่างมีนัยสำคัญ ({utilization_ratio*100:,.1f}%). "
                        f"อาจบ่งชี้ว่าผังนี้ไม่เหมาะสมกับขนาดบ้านที่กำหนด หรือจำเป็นต้องปรับขนาดบ้าน/จำนวนแปลง")
        elif utilization_ratio < 0.8:
            st.info(f"💡 ข้อแนะนำ: พื้นที่รวมจากจำนวนแปลงที่ทำนาย ({total_area_from_units:,.2f} ตร.วา) "
                    f"ต่ำกว่าพื้นที่ขายที่โมเดลทำนายไว้ ({predicted_sale_area:,.2f} ตร.วา) "
                    f"ทำให้มีพื้นที่เหลือค่อนข้างมาก ({((1-utilization_ratio)*100):,.1f}%). "
                    f"อาจสามารถเพิ่มจำนวนแปลงหรือปรับขนาดบ้านให้เหมาะสมยิ่งขึ้นได้")
    else:
        result['ส่วนต่าง (พื้นที่ขาย - พื้นที่รวมแปลง)'] = 0
        result['สัดส่วนพื้นที่แปลงที่ใช้จากพื้นที่ขาย (%)'] = 0
        st.warning("พื้นที่ขายที่ทำนายเป็นศูนย์ ตรวจสอบให้แน่ใจว่าผลการทำนายถูกต้อง")
    
    return result

# --- Streamlit UI: Main App ---
st.title("📐 Smart Layout Predictor (ML Powered)")
st.markdown("โปรดอัปโหลดไฟล์ข้อมูลโครงการของคุณ (layoutdata.xlsx - Sheet1.csv) เพื่อเริ่มต้นการทำนายและวิเคราะห์")

st.markdown("---")
@st.cache_resource
def setup_thai_font():
    try:
        if os.path.exists(THAI_FONT_FILE):
            fm.fontManager.addfont(THAI_FONT_FILE)
            plt.rcParams['font.family'] = fm.FontProperties(fname=THAI_FONT_FILE).get_name()
            st.success(f"ใช้ฟอนต์ '{plt.rcParams['font.family']}' จากไฟล์สำหรับภาษาไทยในกราฟ")
        else:
            thai_fonts = ['Tahoma', 'Sarabun', 'TH SarabunPSK', 'AngsanaUPC', 'CordiaUPC']
            thai_font_path = None
            for font_name in thai_fonts:
                try:
                    font_path_candidate = fm.findfont(fm.FontProperties(family=font_name))
                    if font_path_candidate:
                        thai_font_path = font_path_candidate
                        break
                except Exception:
                    continue
            
            if thai_font_path:
                fm.fontManager.addfont(thai_font_path)
                plt.rcParams['font.family'] = fm.FontProperties(fname=thai_font_path).get_name()
                st.success(f"ใช้ฟอนต์ '{plt.rcParams['font.family']}' ที่พบในระบบสำหรับภาษาไทยในกราฟ")
            else:
                st.warning("ไม่พบฟอนต์ที่รองรับภาษาไทยในระบบ ใช้ฟอนต์เริ่มต้น")
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการตั้งค่าฟอนต์: {e}")
setup_thai_font()

# --- 1. File Uploader and Data Loading ---
uploaded_file = st.file_uploader("เลือกไฟล์ CSV หรือ Excel (Sheet1.csv)", type=["csv", "xlsx"])

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None

df = None
if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)

if df is None:
    st.info("กรุณาอัปโหลดไฟล์ข้อมูลก่อนดำเนินการต่อ")
    st.stop()

# --- Continue only if df is loaded ---
df_original = df.copy()

# --- 2. Data Preprocessing & Model Setup ---
sqm_to_sqwah = 0.25
columns_to_convert = ['พื้นที่โครงการ(ตรม)', 'พื้นที่จัดจำหน่าย(ตรม)', 'พื้นที่สาธา(ตรม)', 'พื้นที่ถนนรวม', 'พื้นที่สวน(5%ของพื้นที่จัดจำหน่าย)']
for col in columns_to_convert:
    if col in df.columns:
        df[col] *= sqm_to_sqwah
        
unit_standard_area_sqm = {
    'ทาวโฮม': 5 * 16,
    'บ้านแฝด': 10 * 16,
    'บ้านเดี่ยว': 15 * 18,
    'บ้านเดี่ยว3ชั้น': 15 * 18,
    'อาคารพาณิชย์': 5 * 16
}
unit_standard_area_sqwah = {
    h_type: area_sqm * sqm_to_sqwah for h_type, area_sqm in unit_standard_area_sqm.items()
}
house_types = ['ทาวโฮม', 'บ้านแฝด', 'บ้านเดี่ยว', 'บ้านเดี่ยว3ชั้น', 'อาคารพาณิชย์']

features = ['พื้นที่โครงการ(ตรม)', 'รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด']
targets = ['พื้นที่จัดจำหน่าย(ตรม)', 'จำนวนหลัง', 'จำนวนซอย', 'พื้นที่ถนนรวม', 'พื้นที่สาธา(ตรม)']
for h_type in house_types:
    if h_type in df.columns:
        targets.append(h_type)

features = [f for f in features if f in df.columns]
targets = [t for t in targets if t in df.columns]

if not features or not targets:
    st.error("ไม่พบ Features หรือ Targets ที่ใช้ในการเทรนโมเดล")
    st.stop()

X = df[features].copy()
y = df[targets].copy()

for col in ['รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด']:
    if col in X.columns and X[col].isnull().any():
        mode_val = X[col].mode()[0]
        X.loc[:, col] = X[col].fillna(mode_val)
for col in ['พื้นที่โครงการ(ตรม)']:
    if col in X.columns and X[col].isnull().any():
        mean_val = X[col].mean()
        X.loc[:, col] = X[col].fillna(mean_val)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = [col for col in ['รูปร่างที่ดิน', 'เกรดโครงการ', 'จังหวัด'] if col in X.columns]
numerical_features = [col for col in ['พื้นที่โครงการ(ตรม)'] if col in X.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- 3. Model Training, Loading, and Tuning ---
st.sidebar.subheader("ตัวเลือกโมเดล")
run_tuning = st.sidebar.checkbox("ต้องการปรับจูนโมเดล (Hyperparameter Tuning) ใหม่?", value=False)
train_button = st.sidebar.button("ฝึก/ปรับจูนโมเดล", key='train_button')

if train_button or (not st.session_state.model_trained and uploaded_file):
    if run_tuning:
        st.sidebar.subheader("กำลังปรับจูน Hyperparameter เพื่อเพิ่มประสิทธิภาพ R² ...")
        with st.spinner('กำลังค้นหา Hyperparameter ที่ดีที่สุด...'):
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))])
            param_grid = {
                'regressor__n_estimators': [100, 200, 300, 400],
                'regressor__max_depth': [5, 10, 15, None],
                'regressor__min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            st.session_state.model = grid_search.best_estimator_
        st.sidebar.success(f"ปรับจูนโมเดลสำเร็จ! Best Parameters: {grid_search.best_params_}")
    else:
        st.sidebar.subheader("กำลังฝึกโมเดล...")
        with st.spinner('กำลังฝึกโมเดล RandomForestRegressor...'):
            regressor = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
            st.session_state.model = Pipeline(steps=[('preprocessor', preprocessor),
                                                     ('regressor', regressor)])
            st.session_state.model.fit(X_train, y_train)
        st.sidebar.success("ฝึกโมเดลสำเร็จ!")
    st.session_state.model_trained = True

# --- 4. Model Evaluation ---
if st.session_state.model_trained:
    model = st.session_state.model
    st.sidebar.subheader("📊 ประสิทธิภาพโมเดล (บนข้อมูลทดสอบ)")
    try:
        y_pred_test = model.predict(X_test)
        st.sidebar.write("**ประสิทธิภาพรวม**")
        st.sidebar.markdown(f"- **MAE เฉลี่ย:** {mean_absolute_error(y_test, y_pred_test):,.2f}")
        st.sidebar.markdown(f"- **R² เฉลี่ย:** {r2_score(y_test, y_pred_test):.4f}")
    except Exception as e:
        st.sidebar.error(f"เกิดข้อผิดพลาดในการประเมินโมเดล: {e}")

# --- 5. User Inputs and Prediction ---
if st.session_state.model_trained:
    st.subheader("📈 ทำนายผังโครงการสำหรับโครงการใหม่")
    st.markdown("ป้อนข้อมูลโครงการเพื่อทำนายผัง, จำนวนแปลง, รายได้, และกำไร")

    with st.expander("สมมติฐานราคาและต้นทุน", expanded=False):
        st.markdown("ค่าเหล่านี้จะใช้สำหรับคำนวณรายได้และกำไร")
        col1, col2, col3 = st.columns(3)
        sale_price_th = col1.number_input("ราคาขาย ทาวน์โฮม (บาท)", value=2_500_000, step=100_000, key='sale_th')
        sale_price_ba = col2.number_input("ราคาขาย บ้านแฝด (บาท)", value=4_000_000, step=100_000, key='sale_ba')
        sale_price_bd = col3.number_input("ราคาขาย บ้านเดี่ยว (บาท)", value=6_000_000, step=100_000, key='sale_bd')
        sale_price_bd3 = st.number_input("ราคาขาย บ้านเดี่ยว3ชั้น (บาท)", value=9_000_000, step=100_000, key='sale_bd3')
        sale_price_comm = st.number_input("ราคาขาย อาคารพาณิชย์ (บาท)", value=5_000_000, step=100_000, key='sale_comm')
        col4, col5, col6 = st.columns(3)
        cost_th = col4.number_input("ต้นทุน ทาวน์โฮม (บาท)", value=1_500_000, step=50_000, key='cost_th')
        cost_ba = col5.number_input("ต้นทุน บ้านแฝด (บาท)", value=2_500_000, step=50_000, key='cost_ba')
        cost_bd = col6.number_input("ต้นทุน บ้านเดี่ยว (บาท)", value=3_500_000, step=50_000, key='cost_bd')
        cost_bd3 = st.number_input("ต้นทุน บ้านเดี่ยว3ชั้น (บาท)", value=5_500_000, step=50_000, key='cost_bd3')
        cost_comm = st.number_input("ต้นทุน อาคารพาณิชย์ (บาท)", value=3_000_000, step=50_000, key='cost_comm')
        col7, col8 = st.columns(2)
        land_cost_per_sqwah = col7.number_input("ต้นทุนที่ดิน (บาท/ตร.วา)", value=50000, step=1000, key='land_cost')
        other_development_cost_ratio = col8.slider("ต้นทุนพัฒนาอื่นๆ (% ของรายได้)", min_value=0.0, max_value=0.3, value=0.20, step=0.01, key='other_cost_ratio')
    sale_price_per_unit = {'ทาวโฮม': sale_price_th, 'บ้านแฝด': sale_price_ba, 'บ้านเดี่ยว': sale_price_bd, 'บ้านเดี่ยว3ชั้น': sale_price_bd3, 'อาคารพาณิชย์': sale_price_comm}
    construction_cost_per_unit = {'ทาวโฮม': cost_th, 'บ้านแฝด': cost_ba, 'บ้านเดี่ยว': cost_bd, 'บ้านเดี่ยว3ชั้น': cost_bd3, 'อาคารพาณิชย์': cost_comm}
    input_col1, input_col2 = st.columns(2)
    project_area_input = input_col1.number_input("พื้นที่โครงการ (ตร.วา)",
                                                  min_value=float(X['พื้นที่โครงการ(ตรม)'].min()),
                                                  max_value=float(X['พื้นที่โครงการ(ตรม)'].max()),
                                                  value=float(X['พื้นที่โครงการ(ตรม)'].mean()),
                                                  step=500.0)
    land_shape_options = X['รูปร่างที่ดิน'].dropna().unique().tolist()
    land_shape_input = input_col2.selectbox("รูปร่างที่ดิน", land_shape_options)
    grade_options = X['เกรดโครงการ'].dropna().unique().tolist()
    grade_input = st.selectbox("เกรดโครงการ", grade_options)
    province_options = X['จังหวัด'].dropna().unique().tolist()
    province_input = st.selectbox("จังหวัด", province_options)

    if st.button("ทำนายผังโครงการ", key='predict_button'):
        with st.spinner('กำลังทำนายผล...'):
            historically_present_types = get_historically_present_house_types(df_original, grade_input)
            result_ml = predict_and_analyze(
                project_area_input, land_shape_input, grade_input, province_input,
                model, house_types, targets, sale_price_per_unit, construction_cost_per_unit,
                land_cost_per_sqwah, other_development_cost_ratio, unit_standard_area_sqwah, historically_present_types
            )
        st.subheader("🔍 ผลการทำนายจาก ML")
        st.dataframe(pd.DataFrame(result_ml.items(), columns=['รายการ', 'ค่าทำนาย']), use_container_width=True)
        st.subheader("📊 การนำเสนอผลลัพธ์ด้วยภาพ")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.write("#### สัดส่วนการแบ่งพื้นที่")
            area_data = [
                result_ml['พื้นที่ขาย (ตร.วา)'],
                result_ml['พื้นที่สาธารณะ (ตร.วา)'],
                result_ml['พื้นที่สวน (ตร.วา)'],
                result_ml['พื้นที่ถนน (ตร.วา)']
            ]
            area_labels = ['พื้นที่ขาย', 'พื้นที่สาธารณะ', 'พื้นที่สวน', 'พื้นที่ถนน']
            plot_area_pie_chart(area_data, area_labels)
        with chart_col2:
            st.write("#### จำนวนแปลงบ้าน")
            plot_house_bar_chart(result_ml, house_types)
        st.subheader("📝 สร้าง Executive Summary อัตโนมัติ")
        if st.button("สร้างสรุปสำหรับผู้บริหารด้วย AI", key='summary_button'):
            with st.spinner('AI กำลังประมวลผล...'):
                summary_prompt = f"""
                คุณเป็นผู้เชี่ยวชาญด้านอสังหาริมทรัพย์ โปรดเขียน Executive Summary กระชับๆ สำหรับผู้บริหาร โดยอิงจากข้อมูลการทำนายโครงการดังนี้:
                - **พื้นที่โครงการทั้งหมด:** {result_ml['พื้นที่โครงการ (ตร.วา)']:,.2f} ตร.วา
                - **พื้นที่ขาย:** {result_ml['พื้นที่ขาย (ตร.วา)']:,.2f} ตร.วา
                - **จำนวนแปลงรวม:** {result_ml['จำนวนแปลงรวม']:,.0f} ยูนิต
                - **แบ่งเป็น:** ทาวน์โฮม {result_ml.get('จำนวนแปลง (ทาวโฮม)', 0):,.0f} หลัง, บ้านแฝด {result_ml.get('จำนวนแปลง (บ้านแฝด)', 0):,.0f} หลัง, บ้านเดี่ยว {result_ml.get('จำนวนแปลง (บ้านเดี่ยว)', 0):,.0f} หลัง
                - **กำไรโดยประมาณ:** {result_ml['กำไร (ทำนาย)']:,.2f} บาท
                - **สัดส่วนพื้นที่แปลงที่ใช้จากพื้นที่ขาย:** {result_ml.get('สัดส่วนพื้นที่แปลงที่ใช้จากพื้นที่ขาย (%)', 0):,.2f}%
                
                โปรดสรุปจุดเด่นของผังโครงการนี้และโอกาสทางธุรกิจในเชิงกลยุทธ์.
                """
                summary_text = call_gemini_api(summary_prompt)
                st.markdown("---")
                st.markdown("#### Executive Summary จาก AI")
                st.write(summary_text)

    # --- 6. Best Option Analysis ---
    st.subheader("💡 วิเคราะห์ทางเลือกที่ดีที่สุด (กำไรสูงสุด)")
    st.markdown("วิเคราะห์เกรดโครงการที่ให้ 'กำไร (ทำนาย)' สูงสุด")
    analysis_land_shape = st.selectbox("รูปร่างที่ดินสำหรับวิเคราะห์", df['รูปร่างที่ดิน'].dropna().unique().tolist(), key='analysis_land_shape')
    analysis_province = st.selectbox("จังหวัดสำหรับวิเคราะห์", df['จังหวัด'].dropna().unique().tolist(), key='analysis_province')
    analysis_project_area = st.number_input("พื้นที่โครงการ (ตร.วา)", min_value=1000.0, value=40000.0, step=500.0, key='analysis_project_area')
    if st.button("ค้นหาทางเลือกที่ดีที่สุด", key='best_option_button'):
        if st.session_state.model_trained:
            with st.spinner('กำลังวิเคราะห์ทางเลือกทั้งหมด...'):
                best_grade_results = []
                available_grades_for_analysis = df['เกรดโครงการ'].dropna().unique().tolist()
                for grade_option in available_grades_for_analysis:
                    historically_present_types = get_historically_present_house_types(df_original, grade_option)
                    predicted_output = predict_and_analyze(
                        analysis_project_area, analysis_land_shape, grade_option, analysis_province,
                        model, house_types, targets, sale_price_per_unit, construction_cost_per_unit,
                        land_cost_per_sqwah, other_development_cost_ratio, unit_standard_area_sqwah, historically_present_types
                    )
                    predicted_output['เกรดโครงการ'] = grade_option
                    best_grade_results.append(predicted_output)
                results_df = pd.DataFrame(best_grade_results)
                if not results_df.empty:
                    results_df_sorted = results_df.sort_values(by='กำไร (ทำนาย)', ascending=False).reset_index(drop=True)
                    st.write("### ผลการทำนายและจัดอันดับตาม 'กำไร (ทำนาย)'")
                    st.dataframe(results_df_sorted, use_container_width=True)
                    best_option = results_df_sorted.iloc[0]
                    st.success(f"**ทางเลือกที่ดีที่สุดคือ: เกรดโครงการ '{best_option['เกรดโครงการ']}'** "
                                f"ซึ่งคาดว่าจะให้ 'กำไร' สูงสุดที่: **{best_option['กำไร (ทำนาย)']:,.2f} บาท**")
                else:
                    st.warning("ไม่สามารถสร้าง DataFrame จากผลลัพธ์ได้")
        else:
            st.warning("กรุณาฝึกโมเดลก่อนวิเคราะห์")

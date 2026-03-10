import streamlit as st
import pandas as pd
import joblib
import re
from datetime import datetime

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI XSMN Predictor", page_icon="🔮", layout="centered")

# CSS để giao diện đẹp hơn trên điện thoại
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #FF4B4B; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- HÀM TẢI DỮ LIỆU ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model_xsmn_predict.pkl')
        df = pd.read_csv('xsmn_final_clean.csv', dtype=str)
        return model, df
    except:
        return None, None

def get_day_lotos(row):
    lotos = []
    # Các cột giải theo định dạng file của bạn
    cols = ['G.8', 'G.7', 'G.6', 'G.5', 'G.4', 'G.3', 'G.2', 'G.1', 'ĐB']
    for col in cols:
        val = str(row.get(col, '')).split('.')[0]
        nums = re.findall(r'\d{2,}', val)
        for n in nums: lotos.append(n[-2:])
    return list(set(lotos))

# --- GIAO DIỆN CHÍNH ---
st.title("🔮 AI Dự Đoán XSMN")
st.info("Hệ thống tự động phân tích nhịp gan và tần suất từ dữ liệu 20 năm.")

model, df = load_assets()

if model is None or df is None:
    st.error("❌ Thiếu file dữ liệu hoặc model! Hãy kiểm tra lại file .csv và .pkl")
else:
    # Chọn Đài
    list_dai = sorted(df['Đài'].unique())
    selected_dai = st.selectbox("🎯 Chọn đài quay thưởng:", list_dai)

    if st.button("🚀 Bắt đầu phân tích & Dự đoán"):
        with st.spinner('Đang tính toán nhịp độ các con số...'):
            # Xử lý dữ liệu đài chọn
            df['Date_DT'] = pd.to_datetime(df['Ngày'], dayfirst=True)
            df_dai = df[df['Đài'] == selected_dai].sort_values('Date_DT').reset_index(drop=True)
            
            history = df_dai.apply(get_day_lotos, axis=1).tolist()
            
            # Tính đặc trưng cho 100 số
            current_features = []
            for n in range(100):
                s_num = f"{n:02d}"
                gap = 0
                for i in range(len(history)-1, -1, -1):
                    if s_num in history[i]: break
                    gap += 1
                f10 = sum(1 for res in history[-10:] if s_num in res)
                f30 = sum(1 for res in history[-30:] if s_num in res)
                was_last = 1 if s_num in history[-1] else 0
                current_features.append([n, gap, f10, f30, was_last])

            X_predict = pd.DataFrame(current_features, columns=['So', 'Gap', 'Freq_10', 'Freq_30', 'Was_Last'])
            
            # Dự đoán xác suất
            probs = model.predict_proba(X_predict)[:, 1]
            X_predict['Xac_Suat'] = probs
            top_3 = X_predict.sort_values('Xac_Suat', ascending=False).head(3).reset_index(drop=True)

            # Hiển thị kết quả
            st.subheader(f"📊 Top 3 tiềm năng - {selected_dai}")
            st.write(f"Cập nhật đến: {df_dai['Ngày'].iloc[-1]}")
            
            cols = st.columns(3)
            for i, row in top_3.iterrows():
                with cols[i]:
                    st.metric(
                        label=f"Top {i+1}", 
                        value=f"{int(row['So']):02d}", 
                        delta=f"{row['Xac_Suat']*100:.1f}%"
                    )
            
            st.balloons()
            st.warning("⚠️ Lưu ý: Kết quả dựa trên xác suất thống kê, chỉ mang tính chất tham khảo.")

st.markdown("---")
st.caption("Phát triển bởi AI XSMN System v2.0")
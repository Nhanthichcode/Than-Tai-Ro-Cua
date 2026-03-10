import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import plotly.express as px

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Thần Tài Dự Đoán - AI XSMN", layout="wide")

# CSS để giao diện đẹp hơn trên điện thoại
st.markdown("""
    <style>
    .result-box { padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid #ddd; }
    .win { background-color: #d4edda; color: #155724; border-left: 5px solid #28a745; }
    .number-circle { 
        display: inline-block; width: 35px; height: 35px; line-height: 35px; 
        border-radius: 50%; background: #f0f2f6; text-align: center; margin: 3px; 
        font-weight: bold; border: 1px solid #ccc;
    }
    .hit { background: #ff4b4b !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. TẢI DỮ LIỆU ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model_xsmn_predict.pkl')
        df = pd.read_csv('xsmn_final_clean.csv', dtype=str)
        df['Date_DT'] = pd.to_datetime(df['Ngày'], dayfirst=True)
        return model, df
    except Exception as e:
        st.error("🚨 Không tìm thấy file dữ liệu hoặc model. Hãy kiểm tra lại GitHub!")
        return None, None

def get_day_lotos(row):
    lotos = []
    for col in ['G.8', 'G.7', 'G.6', 'G.5', 'G.4', 'G.3', 'G.2', 'G.1', 'ĐB']:
        val = str(row.get(col, '')).split('.')[0]
        nums = re.findall(r'\d{2,}', val)
        for n in nums: lotos.append(n[-2:])
    return lotos

model, df = load_assets()

if model and df is not None:
    # --- 3. CHỌN ĐÀI (TỈNH) ---
    st.sidebar.header("📍 Tùy chọn")
    selected_dai = st.sidebar.selectbox("Chọn Đài/Tỉnh:", sorted(df['Đài'].unique()))
    df_dai = df[df['Đài'] == selected_dai].sort_values('Date_DT', ascending=False).reset_index(drop=True)

    tab1, tab2, tab3 = st.tabs(["🔮 Dự Đoán", "📂 Lịch Sử", "📊 Biểu Đồ"])

    # --- TAB 1: DỰ ĐOÁN ---
    with tab1:
        st.subheader(f"Dự đoán kỳ tới đài {selected_dai}")
        if st.button("Bắt đầu soi cầu AI"):
            # Lấy bối cảnh dữ liệu
            history_data = df_dai.head(30)
            draw_history = [get_day_lotos(r) for _, r in history_data.iterrows()]
            
            # Tính Đầu/Đuôi 5 kỳ gần nhất
            flat_5 = [item for sublist in draw_history[:5] for item in sublist]
            h5 = [s[0] for s in flat_5] if flat_5 else []
            t5 = [s[1] for s in flat_5] if flat_5 else []
            
            # --- TÍNH ĐẶC TRƯNG (KHỚP 100% VỚI MODEL) ---
            predict_features = []
            next_thu = (history_data.iloc[0]['Date_DT'].weekday() + 1) % 7 + 2

            for n in range(100):
                s_num = f"{n:02d}"
                gap = 0
                for draw in draw_history:
                    if s_num in draw: break
                    gap += 1
                f5 = sum(1 for d in draw_history[:5] if s_num in d)
                f10 = sum(1 for d in draw_history[:10] if s_num in d)
                f30 = sum(1 for d in draw_history[:30] if s_num in d)
                
                predict_features.append({
                    'So': n, 'Thu': next_thu, 'Gap': gap, 
                    'F5': f5, 'F10': f10, 'F30': f30,
                    'H_Freq': h5.count(s_num[0]), 'T_Freq': t5.count(s_num[1])
                })
            
            # ÉP ĐÚNG THỨ TỰ CỘT
            X_val = pd.DataFrame(predict_features)[['So', 'Thu', 'Gap', 'F5', 'F10', 'F30', 'H_Freq', 'T_Freq']]
            
            # DỰ ĐOÁN
            probs = model.predict_proba(X_val)[:, 1]
            top_3 = [f"{n:02d}" for n in np.argsort(probs)[-3:]]

            st.markdown(f"""<div class="result-box win">
                <h3>💎 Gợi ý Top 3: {', '.join(top_3)}</h3>
                <p>Xác suất nổ được tính toán dựa trên nhịp gan và đầu đuôi 50 kỳ.</p>
            </div>""", unsafe_allow_html=True)

    # --- TAB 2: LỊCH SỬ ---
    with tab2:
        st.subheader("Tra cứu kết quả cũ")
        sel_ngay = st.selectbox("Chọn ngày:", df_dai['Ngày'].tolist())
        row = df_dai[df_dai['Ngày'] == sel_ngay].iloc[0]
        lotos = sorted(get_day_lotos(row))
        
        st.write(f"Kết quả đài {selected_dai} ngày {sel_ngay}:")
        html_lotos = "".join([f'<div class="number-circle">{n}</div>' for n in lotos])
        st.markdown(html_lotos, unsafe_allow_html=True)

    # --- TAB 3: BIỂU ĐỒ ---
    with tab3:
        st.subheader(f"Phân tích xu hướng đài {selected_dai}")
        
        # Thống kê 50 kỳ
        all_lotos = []
        for _, r in df_dai.head(50).iterrows():
            all_lotos.extend(get_day_lotos(r))
        
        df_counts = pd.Series(all_lotos).value_counts().reset_index()
        df_counts.columns = ['Số', 'Lần về']
        
        # Biểu đồ Tần suất
        fig = px.bar(df_counts.head(20), x='Số', y='Lần về', color='Lần về', title="Top 20 số về nhiều nhất")
        st.plotly_chart(fig, use_container_width=True)
        
        # Bản đồ nhiệt Đầu - Đuôi
        heatmap_data = np.zeros((10, 10))
        for n in range(100):
            heatmap_data[n//10, n%10] = all_lotos.count(f"{n:02d}")
        
        fig_heat = px.imshow(heatmap_data, text_auto=True, title="Bản đồ nhiệt Đầu - Đuôi",
                            labels=dict(x="Đuôi", y="Đầu", color="Số lần"),
                            x=[str(i) for i in range(10)], y=[str(i) for i in range(10)])
        st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.info("👋 Hãy đảm bảo bạn đã đẩy file model.pkl và csv lên GitHub!")
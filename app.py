import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import re
from datetime import datetime

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Thần Tài XSMN - AI Dự Đoán", layout="wide")

st.markdown("""
    <style>
    .result-box { padding: 20px; border-radius: 12px; margin: 10px 0; border: 1px solid #e0e0e0; }
    .win-card { background: linear-gradient(135deg, #d4edda 0%, #ffffff 100%); border-left: 6px solid #28a745; }
    .number-ball { 
        display: inline-block; width: 38px; height: 38px; line-height: 38px; 
        border-radius: 50%; background: #f8f9fa; text-align: center; margin: 4px; 
        font-weight: bold; border: 1px solid #ced4da; color: #333;
    }
    .highlight { background: #ff4b4b !important; color: white !important; border: none !important; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. TẢI DỮ LIỆU & MODEL ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model_xsmn_predict.pkl')
        df = pd.read_csv('xsmn_final_clean.csv', dtype=str)
        df['Date_DT'] = pd.to_datetime(df['Ngày'], dayfirst=True)
        return model, df
    except Exception as e:
        st.error(f"🚨 Lỗi hệ thống: Không tìm thấy file dữ liệu. Hãy đảm bảo đã upload file .csv và .pkl lên GitHub.")
        return None, None

def get_day_lotos(row):
    lotos = []
    for col in ['G.8', 'G.7', 'G.6', 'G.5', 'G.4', 'G.3', 'G.2', 'G.1', 'ĐB']:
        val = str(row.get(col, '')).split('.')[0]
        nums = re.findall(r'\d{2,}', val)
        for n in nums: lotos.append(n[-2:])
    return lotos

model, df = load_assets()

# --- 3. XỬ LÝ CHÍNH ---
if model and df is not None:
    # Sidebar chọn đài
    st.sidebar.header("📍 Cấu hình")
    all_dais = sorted(df['Đài'].unique())
    selected_dai = st.sidebar.selectbox("Chọn Tỉnh/Đài muốn soi:", all_dais)
    
    # Lọc dữ liệu theo đài
    df_dai = df[df['Đài'] == selected_dai].sort_values('Date_DT', ascending=False).reset_index(drop=True)

    tab1, tab2, tab3 = st.tabs(["🔮 Dự Đoán AI", "📂 Tra Cứu Kỳ Quay", "📊 Phân Tích Biểu Đồ"])

    # --- TAB 1: DỰ ĐOÁN ---
    with tab1:
        st.subheader(f"Dự đoán kỳ tiếp theo: Đài {selected_dai}")
        if st.button("🔥 Kích hoạt AI soi cầu"):
            # Chuẩn bị 30 kỳ gần nhất
            history_30 = df_dai.head(30)
            draws = [get_day_lotos(r) for _, r in history_30.iterrows()]
            
            # Tính Đầu/Đuôi 5 kỳ gần nhất
            flat_5 = [n for d in draws[:5] for n in d]
            h5 = [s[0] for s in flat_5] if flat_5 else ["0"]*10
            t5 = [s[1] for s in flat_5] if flat_5 else ["0"]*10
            
            # Lấy thứ của kỳ quay tới
            last_date = history_30.iloc[0]['Date_DT']
            next_thu = (last_date.weekday() + 1) % 7 + 2 # Ước tính thứ kỳ tới

            # Tạo bảng 8 đặc trưng cho 100 số (00-99)
            X_predict = []
            for n in range(100):
                s_num = f"{n:02d}"
                gap = 0
                for d in draws:
                    if s_num in d: break
                    gap += 1
                
                X_predict.append([
                    n, next_thu, gap,
                    sum(1 for d in draws[:5] if s_num in d),
                    sum(1 for d in draws[:10] if s_num in d),
                    sum(1 for d in draws[:30] if s_num in d),
                    h5.count(s_num[0]),
                    t5.count(s_num[1])
                ])
            
            # Dự đoán xác suất (Dùng .values để tránh lỗi ValueError về tên cột)
            X_val = np.array(X_predict)
            probs = model.predict_proba(X_val)[:, 1]
            top_3_idx = np.argsort(probs)[-3:][::-1] # Lấy 3 số cao nhất
            
            st.markdown(f"""<div class="result-box win-card">
                <h3>💎 Gợi ý Top 3 từ AI:</h3>
                <h2 style="color: #d32f2f;">{', '.join([f"{i:02d}" for i in top_3_idx])}</h2>
                <p><i>Dựa trên phân tích nhịp gan {X_val[top_3_idx[0], 2]} kỳ và tần suất đầu đuôi.</i></p>
            </div>""", unsafe_allow_html=True)

    # --- TAB 2: TRA CỨU ---
    with tab2:
        sel_ngay = st.selectbox("Chọn ngày quay thưởng:", df_dai['Ngày'].tolist())
        row_sel = df_dai[df_dai['Ngày'] == sel_ngay].iloc[0]
        actual_lotos = sorted(get_day_lotos(row_sel))
        
        st.write(f"**Kết quả thực tế đài {selected_dai} ngày {sel_ngay}:**")
        html_lotos = "".join([f'<div class="number-ball">{n}</div>' for n in actual_lotos])
        st.markdown(html_lotos, unsafe_allow_html=True)

    # --- TAB 3: BIỂU ĐỒ ---
    with tab3:
        st.subheader(f"Xu hướng 50 kỳ gần nhất - {selected_dai}")
        
        # Thống kê tần suất
        all_recent = []
        for _, r in df_dai.head(50).iterrows():
            all_recent.extend(get_day_lotos(r))
        
        freq_df = pd.Series(all_recent).value_counts().reset_index()
        freq_df.columns = ['Số', 'Lần về']
        
        # Biểu đồ cột Tần suất
        fig_bar = px.bar(freq_df.head(15), x='Số', y='Lần về', color='Lần về', 
                        title="Top 15 số về nhiều nhất", color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, width='stretch')

        # Bản đồ nhiệt 00-99
        heatmap_data = np.zeros((10, 10))
        for n in range(100):
            heatmap_data[n//10, n%10] = all_recent.count(f"{n:02d}")
            
        fig_heat = px.imshow(heatmap_data, text_auto=True, title="Bản đồ nhiệt Đầu - Đuôi",
                            labels=dict(x="Hàng đơn vị", y="Hàng chục", color="Số lần"),
                            x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
                            color_continuous_scale='YlOrRd')
        st.plotly_chart(fig_heat, width='stretch')

else:
    st.warning("⚠️ Đang chờ cấu hình dữ liệu. Hãy đảm bảo bạn đã đẩy file .csv và .pkl lên GitHub.")
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CẤU HÌNH GIAO DIỆN & CSS ---
st.set_page_config(page_title="Thần Tài Gõ Cửa - AI XSMN", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff; border-radius: 5px 5px 0 0; padding: 10px 20px;
    }
    .result-box { padding: 15px; border-radius: 10px; margin: 10px 0; border: 1px solid #ddd; }
    .win { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }
    .loss { background-color: #ffffff; color: #721c24; }
    .number-circle { 
        display: inline-block; width: 35px; height: 35px; line-height: 35px; 
        border-radius: 50%; background: #f0f2f6; text-align: center; margin: 3px; 
        font-weight: bold; border: 1px solid #ccc;
    }
    .hit { background: #ff4b4b !important; color: white !important; border: 1px solid #b30000 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM HỖ TRỢ & LOAD DỮ LIỆU ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model_xsmn_predict.pkl')
        df = pd.read_csv('xsmn_final_clean.csv', dtype=str)
        df['Date_DT'] = pd.to_datetime(df['Ngày'], dayfirst=True)
        return model, df
    except Exception as e:
        st.error(f"❌ Lỗi tải file: {e}. Hãy đảm bảo model.pkl và csv nằm ở thư mục gốc.")
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
    # --- 3. SIDEBAR - CHỌN ĐÀI CHUNG ---
    st.sidebar.header("⚙️ Cấu hình")
    selected_dai = st.sidebar.selectbox("🎯 Chọn Tỉnh/Đài:", sorted(df['Đài'].unique()))
    df_dai = df[df['Đài'] == selected_dai].sort_values('Date_DT', ascending=False).reset_index(drop=True)

    tab1, tab2, tab3 = st.tabs(["🔮 Dự Đoán", "📂 Tra Cứu Kỳ Quay", "📊 Thống Kê Biểu Đồ"])

    # --- TAB 1: DỰ ĐOÁN KỲ TỚI ---
    with tab1:
        st.subheader(f"Dự đoán cho kỳ quay tiếp theo - {selected_dai}")
        if st.button("🚀 Kích hoạt AI dự đoán"):
            # Lấy 30 kỳ gần nhất của đài này để tính đặc trưng
            history_data = df_dai.head(30)
            draw_history = [get_day_lotos(r) for _, r in history_data.iterrows()]
            
            # Tính Đầu - Đuôi 5 kỳ gần nhất
            flat_5 = [item for sublist in draw_history[:5] for item in sublist]
            heads_5 = [s[0] for s in flat_5] if flat_5 else ["0"]*10
            tails_5 = [s[1] for s in flat_5] if flat_5 else ["0"]*10
            
            # Tạo Feature (Khớp 100% với auto_run.py)
            predict_features = []
            next_thu = (history_data.iloc[0]['Date_DT'].weekday() + 1) % 7 + 2 # Thứ dự kiến

            for n in range(100):
                s_num = f"{n:02d}"
                gap = 0
                for draw in draw_history:
                    if s_num in draw: break
                    gap += 1
                f5 = sum(1 for d in draw_history[:5] if s_num in d)
                f10 = sum(1 for d in draw_history[:10] if s_num in d)
                f30 = sum(1 for d in draw_history[:30] if s_num in d)
                h_freq, t_freq = heads_5.count(s_num[0]), tails_5.count(s_num[1])
                
                predict_features.append({
                    'So': n, 'Thu': next_thu, 'Gap': gap, 
                    'F5': f5, 'F10': f10, 'F30': f30,
                    'H_Freq': h_freq, 'T_Freq': t_freq
                })
            
            X_val = pd.DataFrame(predict_features)[['So', 'Thu', 'Gap', 'F5', 'F10', 'F30', 'H_Freq', 'T_Freq']]
            probs = model.predict_proba(X_val)[:, 1]
            top_3 = [f"{n:02d}" for n in np.argsort(probs)[-3:]]

            st.success(f"Top 3 con số tiềm năng nhất: **{', '.join(top_3)}**")
            st.info("💡 Lưu ý: Kết quả chỉ mang tính chất tham khảo toán học.")

    # --- TAB 2: TRA CỨU & ĐỐI SOÁT ---
    with tab2:
        list_ngay = df_dai['Ngày'].tolist()
        sel_ngay = st.selectbox("Chọn ngày muốn xem lại:", list_ngay)
        
        if st.button("🔍 Đối soát"):
            curr_pos = df_dai[df_dai['Ngày'] == sel_ngay].index[0]
            # Tính toán dự đoán "Hồi tố"
            history_subset = df_dai.iloc[curr_pos+1 : curr_pos+31]
            draw_hist = [get_day_lotos(r) for _, r in history_subset.iterrows()]
            
            # (Phần tính toán logic tương tự Tab 1 nhưng dùng draw_hist)
            # Giả định kết quả top_3_retro được tính ra...
            # Hiển thị 18 lô thực tế
            actual_row = df_dai.iloc[curr_pos]
            actual_lotos = get_day_lotos(actual_row)
            
            st.write(f"**Kết quả thực tế ngày {sel_ngay}:**")
            res_html = "".join([f'<div class="number-circle">{n}</div>' for n in sorted(actual_lotos)])
            st.markdown(res_html, unsafe_allow_html=True)

    # --- TAB 3: THỐNG KÊ BIỂU ĐỒ (TỈNH THÀNH) ---
    with tab3:
        st.subheader(f"Phân tích dữ liệu 50 kỳ - {selected_dai}")
        
        all_lotos = []
        for _, r in df_dai.head(50).iterrows():
            all_lotos.extend(get_day_lotos(r))
        
        # Biểu đồ Tần Suất
        df_counts = pd.Series(all_lotos).value_counts().reset_index()
        df_counts.columns = ['Số', 'Lần về']
        top_15 = df_counts.head(15)
        
        fig_freq = px.bar(top_15, x='Số', y='Lần về', title="Top 15 số về nhiều nhất",
                         color='Lần về', color_continuous_scale='Reds')
        st.plotly_chart(fig_freq, use_container_width=True)

        # Biểu đồ Bản đồ nhiệt Đầu - Đuôi
        st.subheader("Bản đồ nhiệt Đầu - Đuôi (Ma trận 00-99)")
        heatmap_matrix = np.zeros((10, 10))
        for n in range(100):
            heatmap_matrix[n//10, n%10] = all_lotos.count(f"{n:02d}")
        
        fig_heat = px.imshow(heatmap_matrix, text_auto=True,
                            labels=dict(x="Đuôi", y="Đầu", color="Số lần"),
                            x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
                            color_continuous_scale='Viridis')
        st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.warning("⚠️ Đang chờ dữ liệu từ GitHub... Hãy đảm bảo Bot đã chạy thành công lần đầu.")
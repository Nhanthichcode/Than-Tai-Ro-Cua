import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI XSMN - Tra Cứu Kỳ Quay", layout="wide")

st.markdown("""
    <style>
    .result-box { padding: 20px; border-radius: 10px; margin: 10px 0; }
    .win { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
    .loss { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
    .number-circle { 
        display: inline-block; width: 40px; height: 40px; line-height: 40px; 
        border-radius: 50%; background: #eee; text-align: center; margin: 2px; font-weight: bold;color: #000;
    }
    .hit { background: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- HÀM HỖ TRỢ ---
@st.cache_resource
def load_assets():
    model = joblib.load('model_xsmn_predict.pkl')
    df = pd.read_csv('xsmn_final_clean.csv', dtype=str)
    df['Date_DT'] = pd.to_datetime(df['Ngày'], dayfirst=True)
    return model, df

def get_day_lotos(row):
    lotos = []
    for col in ['G.8', 'G.7', 'G.6', 'G.5', 'G.4', 'G.3', 'G.2', 'G.1', 'ĐB']:
        val = str(row.get(col, '')).split('.')[0]
        nums = re.findall(r'\d{2,}', val)
        for n in nums: lotos.append(n[-2:])
    return lotos

model, df = load_assets()

# --- GIAO DIỆN CHÍNH ---
st.title("📂 Tra cứu Dự đoán & Kết quả theo Kỳ")

# 1. Bộ lọc chọn Đài và Ngày
col_select1, col_select2 = st.columns(2)

with col_select1:
    selected_dai = st.selectbox("🎯 Chọn Đài:", sorted(df['Đài'].unique()))

# Lọc danh sách ngày của đài đó
df_dai = df[df['Đài'] == selected_dai].sort_values('Date_DT', ascending=False)
list_ngay = df_dai['Ngày'].tolist()

with col_select2:
    selected_ngay = st.selectbox("📅 Chọn Kỳ quay (Ngày):", list_ngay)

if st.button("🔍 Xem chi tiết kỳ quay này"):
    # Lấy index của ngày được chọn
    target_idx = df_dai[df_dai['Ngày'] == selected_ngay].index[0]
    # Lấy vị trí dòng trong df_dai đã sắp xếp
    current_pos = df_dai.index.get_loc(target_idx)
    
    # Kiểm tra nếu có đủ dữ liệu quá khứ để dự đoán (cần ít nhất 30 kỳ trước đó)
    if current_pos >= len(df_dai) - 30:
        st.warning("⚠️ Dữ liệu quá khứ không đủ để AI đưa ra dự đoán chính xác cho kỳ này.")
    else:
        # A. LOGIC DỰ ĐOÁN (Chỉ dùng dữ liệu TRƯỚC ngày được chọn)
        # Lấy lịch sử từ vị trí hiện tại trở về trước (về quá khứ)
        history_data = df_dai.iloc[current_pos+1 : current_pos+31] # 30 kỳ trước
        draw_history = [get_day_lotos(r) for _, r in history_data.iterrows()]
        
        # Tính đặc trưng (Gap, Freq...)
        predict_features = []
        for n in range(100):
            s_num = f"{n:02d}"
            gap = 0
            for draw in draw_history:
                if s_num in draw: break
                gap += 1
            f10 = sum(1 for d in draw_history[:10] if s_num in d)
            f30 = sum(1 for d in draw_history[:30] if s_num in d)
            predict_features.append([n, gap, f10, f30])
        
        X_val = pd.DataFrame(predict_features, columns=['So', 'Gap', 'F10', 'F30'])
        probs = model.predict_proba(X_val)[:, 1]
        top_3_idx = np.argsort(probs)[-3:]
        top_3_nums = [f"{n:02d}" for n in top_3_idx]

        # B. KẾT QUẢ THỰC TẾ
        actual_row = df_dai.iloc[current_pos]
        actual_lotos = get_day_lotos(actual_row)
        
        # C. HIỂN THỊ SO SÁNH
        st.markdown("---")
        col_pred, col_act = st.columns(2)
        
        with col_pred:
            st.subheader("🔮 AI Dự đoán (Top 3)")
            is_win = any(n in actual_lotos for n in top_3_nums)
            status_class = "win" if is_win else "loss"
            st.markdown(f"""<div class="result-box {status_class}">
                <b>Số gợi ý:</b> {', '.join(top_3_nums)} <br>
                <b>Trạng thái:</b> {'✅ TRÚNG' if is_win else '❌ TRƯỢT'}
            </div>""", unsafe_allow_html=True)

        with col_act:
            st.subheader("📜 Kết quả thực tế (18 lô)")
            loto_html = ""
            for n in sorted(set(actual_lotos)):
                hit_class = "hit" if n in top_3_nums else ""
                loto_html += f'<div class="number-circle {hit_class}">{n}</div>'
            st.markdown(loto_html, unsafe_allow_html=True)
            st.caption("Các số màu đỏ là số AI đã dự đoán đúng.")

        # D. CHI TIẾT CÁC GIẢI
        with st.expander("Xem bảng bảng giải chi tiết"):
            st.table(pd.DataFrame([actual_row[['G.8','G.7','G.6','G.5','G.4','G.3','G.2','G.1','ĐB']]]))
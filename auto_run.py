import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
import re
import os

# --- CẤU HÌNH ---
DATA_FILE = 'xsmn_final_clean.csv'
MODEL_FILE = 'model_xsmn_predict.pkl'
PIVOT_DATE = pd.to_datetime('2009-04-01')

# ==========================================
# 1. BỘ CÀO DỮ LIỆU TỰ ĐỘNG (SCRAPER)
# ==========================================
def scrape_today_xsmn():
    print("🌐 Đang kết nối để lấy kết quả hôm nay...")
    url = "https://www.minhngoc.com.vn/ket-qua-xo-so/mien-nam.html"
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Tìm bảng kết quả
        table = soup.find('table', class_='bkqmiennam')
        if not table: return None
        
        date_str = soup.find('td', class_='ngay').text.strip() # Định dạng dd/mm/yyyy
        
        # Tìm các đài quay thưởng hôm nay
        dais = [td.text.strip() for td in table.find_all('td', class_='tinh')]
        
        new_data = []
        # Duyệt qua từng đài để lấy các giải
        for i, dai in enumerate(dais):
            row_data = {'Ngày': date_str, 'Đài': dai}
            # Tìm các giải từ G8 đến ĐB của đài đó
            for g in range(8, 0, -1):
                prize_name = f'G.{g}'
                prize_cells = table.find_all('td', class_=f'g{g}')
                # Lấy cụm số của đài thứ i
                val = prize_cells[i].text.strip().replace("\n", "")
                row_data[prize_name] = val
            
            # Giải Đặc Biệt
            db_cells = table.find_all('td', class_='db')
            row_data['ĐB'] = db_cells[i].text.strip()
            new_data.append(row_data)
            
        print(f"✅ Đã cào xong dữ liệu ngày {date_str} cho {len(dais)} đài.")
        return pd.DataFrame(new_data)
    except Exception as e:
        print(f"❌ Lỗi khi cào dữ liệu: {e}")
        return None

# ==========================================
# 2. BỘ XỬ LÝ & BĂM SỐ (DATA PROCESSING)
# ==========================================
def get_day_lotos(row):
    lotos = []
    curr_date = pd.to_datetime(row['Ngày'], dayfirst=True)
    config = {'G.8':(1,2), 'G.7':(1,3), 'G.6':(3,4), 'G.5':(1,4), 
              'G.4':(7,5), 'G.3':(2,5), 'G.2':(1,5), 'G.1':(1,5), 'ĐB':(1,6)}
    for col, (count, length) in config.items():
        val = str(row.get(col, '')).split('.')[0]
        actual_len = 5 if (col == 'ĐB' and curr_date < PIVOT_DATE) else length
        val = val.zfill(count * actual_len)
        for i in range(count):
            s = val[i*actual_len : (i+1)*actual_len]
            if len(s) >= 2: lotos.append(s[-2:])
    return lotos

# ==========================================
# 3. BỘ HUẤN LUYỆN AI NÂNG CAO (OPTIMIZED TRAINING)
# ==========================================
def train_brain(df):
    print("🧠 Đang tái huấn luyện bộ não AI (Bản nâng cấp 10-20MB)...")
    df['Date_DT'] = pd.to_datetime(df['Ngày'], dayfirst=True)
    ml_rows = []
    
    # Chỉ lấy dữ liệu của 5 đài phổ biến nhất để huấn luyện nhanh và thông minh
    popular_dais = df['Đài'].value_counts().head(5).index
    
    for dai in popular_dais:
        group = df[df['Đài'] == dai].sort_values('Date_DT').reset_index(drop=True)
        draws = group.apply(get_day_lotos, axis=1).tolist()
        
        # Chỉ học 100 kỳ gần nhất của mỗi đài để bộ não luôn "nhạy bén"
        start_idx = max(50, len(group) - 150)
        for i in range(start_idx, len(group)):
            for n in range(100):
                s_num = f"{n:02d}"
                gap = 0
                for j in range(i-1, -1, -1):
                    if s_num in draws[j]: break
                    gap += 1
                f10 = sum(1 for d in draws[i-10:i] if s_num in d)
                f30 = sum(1 for d in draws[i-30:i] if s_num in d)
                ml_rows.append([n, gap, f10, f30, 1 if s_num in draws[i] else 0])

    train_df = pd.DataFrame(ml_rows, columns=['So', 'Gap', 'F10', 'F30', 'Target'])
    
    # --- THAY ĐỔI ĐỘ SÂU ĐỂ SỬA LỖI 1MB ---
    model = RandomForestClassifier(
        n_estimators=150,  # Tăng số cây
        max_depth=18,       # Tăng độ sâu (vừa đủ để thông minh, không quá nặng)
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_df.drop(columns=['Target']), train_df['Target'])
    
    # Lưu và nén mức độ 1 để giữ chất lượng
    joblib.dump(model, MODEL_FILE, compress=1)
    print(f"✅ Bộ não mới đã sẵn sàng! Dung lượng file: {os.path.getsize(MODEL_FILE)/1024:.2f} KB")

# ==========================================
# 4. CHẠY TỰ ĐỘNG (MAIN)
# ==========================================
if __name__ == "__main__":
    # Bước 1: Cập nhật dữ liệu
    df_old = pd.read_csv(DATA_FILE)
    df_new = scrape_today_xsmn()
    
    if df_new is not None:
        # Gộp và xóa trùng lặp dựa trên Ngày + Đài
        df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=['Ngày', 'Đài'], keep='last')
        df_combined.to_csv(DATA_FILE, index=False)
        print("💾 Đã cập nhật file CSV thành công.")
        
        # Bước 2: Dạy lại AI với dữ liệu mới
        train_brain(df_combined)
    else:
        print("⚠️ Không lấy được dữ liệu mới hôm nay. Có thể chưa đến giờ quay.")

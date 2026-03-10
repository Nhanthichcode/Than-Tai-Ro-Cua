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
def scrape_today_xsmn():
    print("🌐 Đang kết nối để lấy kết quả mới nhất...")
    url = "https://www.minhngoc.com.vn/ket-qua-xo-so/mien-nam.html"
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table', class_='bkqmiennam')
        if not table: 
            print("⚠️ Không tìm thấy bảng kết quả.")
            return None
        
        # Lấy ngày hiện tại trên web
        date_node = soup.find('td', class_='ngay')
        if not date_node: return None
        date_str = date_node.text.strip()
        
        # Lấy danh sách Đài
        dais = [td.text.strip() for td in table.find_all('td', class_='tinh')]
        num_dais = len(dais)
        if num_dais == 0: return None
        
        new_data = []
        for i in range(num_dais):
            row_data = {'Ngày': date_str, 'Đài': dais[i]}
            
            # Kiểm tra an toàn cho từng giải
            for g in range(1, 9):
                prize_name = f'G.{g}'
                cells = table.find_all('td', class_=f'g{g}')
                
                # CHỐT CHẶN LỖI: Kiểm tra xem có đủ cột cho đài này không
                if len(cells) > i:
                    row_data[prize_name] = cells[i].text.strip().replace("\n", "")
                else:
                    row_data[prize_name] = "" # Nếu chưa có số thì để trống
            
            # Giải Đặc biệt
            db_cells = table.find_all('td', class_='db')
            row_data['ĐB'] = db_cells[i].text.strip() if len(db_cells) > i else ""
            
            # Chỉ thêm vào nếu đài này đã có đủ số (tránh lấy đài đang quay dở)
            if row_data['ĐB'] != "":
                new_data.append(row_data)
            
        return pd.DataFrame(new_data) if new_data else None
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")
        return None
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

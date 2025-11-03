# HƯỚNG DẪN SỬ DỤNG ỨNG DỤNG PHÂN TÍCH THỊ TRƯỜNG CHỨNG KHOÁN

**Môn học:** DAP391m - Data Analytics Project  
**Học kỳ:** Fall 2025  
**Ngày cập nhật:** 03/11/2025

---

## TỔNG QUAN ỨNG DỤNG

Đây là ứng dụng web phân tích thị trường chứng khoán được xây dựng bằng Python và Streamlit, tích hợp Machine Learning và trí tuệ nhân tạo để hỗ trợ nhà đầu tư đưa ra quyết định giao dịch thông minh.

### CÔNG NGHỆ SỬ DỤNG
- **Ngôn ngữ:** Python 3.11+
- **Framework:** Streamlit (Web application)
- **Dữ liệu:** Yahoo Finance API (yfinance)
- **Machine Learning:** Prophet, Random Forest, LSTM
- **AI Chatbot:** OpenAI GPT-4 + Google Gemini 2.0
- **Visualization:** Plotly, Matplotlib, Folium

---

## CẤU TRÚC ỨNG DỤNG

Ứng dụng có **6 TAB chính**:

### 1. PHÂN TÍCH KỸ THUẬT (Technical Analysis)
**Chức năng:**
- Hiển thị biểu đồ nến (Candlestick chart) của cổ phiếu
- Tính toán và hiển thị các chỉ báo kỹ thuật:
  - RSI (Relative Strength Index): Đo sức mạnh xu hướng
  - MACD (Moving Average Convergence Divergence): Phát hiện tín hiệu mua/bán
  - Bollinger Bands: Đo biến động giá
  - Moving Averages (MA20, MA50, MA200): Đường trung bình động

**Cách sử dụng:**
1. Chọn mã cổ phiếu (VD: AAPL, GOOGL, MSFT)
2. Chọn khoảng thời gian phân tích
3. Xem biểu đồ và các chỉ báo kỹ thuật
4. Đọc phân tích tự động về xu hướng giá

### 2. PHÂN TÍCH CƠ BẢN (Fundamental Analysis)
**Chức năng:**
- Hiển thị thông tin công ty
- Các chỉ số tài chính quan trọng:
  - P/E Ratio (Price to Earnings): Tỷ lệ giá/thu nhập
  - Market Cap: Vốn hóa thị trường
  - Revenue & Profit: Doanh thu và lợi nhuận
  - EPS (Earnings Per Share): Thu nhập mỗi cổ phiếu

**Cách sử dụng:**
1. Chọn mã cổ phiếu
2. Xem thông tin tổng quan công ty
3. Phân tích các chỉ số tài chính
4. So sánh với các công ty cùng ngành

### 3. DỰ ĐOÁN GIÁ (Price Prediction)
**Chức năng:**
- Dự đoán giá cổ phiếu trong tương lai sử dụng 3 mô hình AI:
  - **Prophet**: Dự đoán xu hướng dài hạn
  - **Random Forest**: Phân tích dựa trên nhiều yếu tố
  - **LSTM**: Mạng nơ-ron sâu học từ dữ liệu lịch sử

**Cách sử dụng:**
1. Chọn mã cổ phiếu
2. Chọn mô hình dự đoán (Prophet/Random Forest/LSTM)
3. Chọn số ngày cần dự đoán (1-30 ngày)
4. Xem biểu đồ dự đoán và độ chính xác mô hình

**Lưu ý:** Kết quả dự đoán chỉ mang tính chất tham khảo, không phải lời khuyên đầu tư.

### 4. TRỰC QUAN HÓA NÂNG CAO (Advanced Visualization)
**Chức năng:**
- **Biểu đồ Choropleth Map**: Bản đồ thế giới hiển thị hiệu suất thị trường các quốc gia
- **Biểu đồ 3D**: Phân tích tương quan giữa các chỉ số
- **Word Cloud**: Đám mây từ khóa về thị trường
- **Distribution Plot**: Phân bố lợi nhuận

**Cách sử dụng:**
1. Chọn loại biểu đồ muốn xem
2. Tùy chỉnh các thông số (nếu có)
3. Xem và phân tích biểu đồ tương tác
4. Xuất hình ảnh nếu cần

### 5. GIẢI THÍCH AI (Explainable AI)
**Chức năng:**
- Giải thích chi tiết cách mô hình AI đưa ra dự đoán
- Hiển thị tầm quan trọng của từng yếu tố ảnh hưởng đến giá
- 4 loại phân tích:
  - **Model Analysis Demo**: Tổng quan về mô hình
  - **Feature Importance**: Mức độ quan trọng các yếu tố
  - **Prediction Explanation**: Giải thích dự đoán cụ thể
  - **Model Comparison**: So sánh các mô hình

**Cách sử dụng:**
1. Chọn loại phân tích
2. Kéo thanh trượt để chọn điểm dữ liệu cần giải thích
3. Xem biểu đồ SHAP Waterfall
4. Đọc giải thích chi tiết bằng văn bản

### 6. CHATBOT AI (AI Assistant)
**Chức năng:**
- Trợ lý AI thông minh trả lời câu hỏi về:
  - Phân tích cổ phiếu cụ thể
  - Giải thích các chỉ báo kỹ thuật
  - Tư vấn chiến lược đầu tư
  - Giải đáp thắc mắc về thị trường

**3 chế độ hoạt động:**
- **Single Response**: Câu trả lời từ 1 AI (nhanh)
- **Comparison Mode**: So sánh câu trả lời của Gemini vs GPT-4
- **Enhanced Analysis**: Phân tích chuyên sâu

**Cách sử dụng:**
1. Chọn chế độ chatbot
2. Nhập câu hỏi vào ô chat (VD: "RSI là gì?", "Phân tích AAPL")
3. Đợi AI trả lời (1-3 giây)
4. Tiếp tục hội thoại để làm rõ thêm

---

## HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY ỨNG DỤNG

### YÊU CẦU HỆ THỐNG
- macOS / Windows / Linux
- Python 3.11 trở lên
- RAM: Tối thiểu 4GB, khuyến nghị 8GB
- Dung lượng: 2GB cho cache và models
- Kết nối internet (để tải dữ liệu real-time)

### BƯỚC 1: CÀI ĐẶT DEPENDENCIES
```bash
# Di chuyển vào thư mục project
cd /Users/vudjeuvuj84gmail.com/Downloads/STUDY/FPTU/2025/DAP391m/Project

# Kích hoạt virtual environment
source .venv/bin/activate

# Cài đặt các thư viện (nếu chưa cài)
pip install -r requirements.txt
```

### BƯỚC 2: CẤU HÌNH API KEYS (Tùy chọn)
Nếu muốn sử dụng AI Chatbot, cần cấu hình API keys:

1. Tạo file `.env` trong thư mục project
2. Thêm các dòng sau:
```
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
```

**Lưu ý:** Ứng dụng vẫn chạy được mà không cần API keys, chỉ chatbot không hoạt động.

### BƯỚC 3: CHẠY ỨNG DỤNG
```bash
# Lệnh đầy đủ (khuyến nghị)
source .venv/bin/activate && pkill -f streamlit; sleep 2; streamlit run src/streamlit_app.py --server.port 8507

# Hoặc lệnh ngắn gọn (nếu đã activate)
streamlit run src/streamlit_app.py --server.port 8507
```

### BƯỚC 4: MỞ TRÌNH DUYỆT
Sau khi chạy lệnh, terminal sẽ hiển thị:
```
Local URL: http://localhost:8507
Network URL: http://10.1.159.194:8507
```

Mở trình duyệt và truy cập: **http://localhost:8507**

### BƯỚC 5: ĐĂNG NHẬP
- **Username:** demo
- **Password:** demo123

---

## CÁC TÍNH NĂNG NỔI BẬT

### 1. DEMO MODE
- Ứng dụng có chế độ demo với dữ liệu tổng hợp
- Không cần kết nối internet để test
- Phù hợp cho việc trình bày và học tập

### 2. REAL-TIME DATA
- Tải dữ liệu thực tế từ Yahoo Finance
- Cập nhật tự động mỗi 1 phút
- Hỗ trợ hơn 1000+ mã cổ phiếu toàn cầu

### 3. CACHING THÔNG MINH
- Lưu cache dữ liệu để tăng tốc độ
- Giảm số lần gọi API
- Tối ưu hiệu suất cho nhiều người dùng

### 4. RESPONSIVE DESIGN
- Giao diện thân thiện, dễ sử dụng
- Tự động điều chỉnh theo kích thước màn hình
- Hỗ trợ cả desktop và tablet

---

## CẤU TRÚC DỰ ÁN

```
Project/
├── src/                              # Mã nguồn chính
│   ├── streamlit_app.py              # Ứng dụng chính (1473 dòng)
│   ├── data_loader.py                # Tải và xử lý dữ liệu
│   ├── technical_indicators.py       # Tính toán chỉ báo kỹ thuật
│   ├── prediction_models.py          # Các mô hình dự đoán AI
│   ├── explainable_ai.py             # Giải thích AI
│   ├── chatbot_enhanced.py           # AI Chatbot (743 dòng)
│   ├── advanced_visualization.py     # Trực quan hóa nâng cao
│   └── auth_manager.py               # Quản lý xác thực
├── data/                             # Dữ liệu
│   └── demo_data/                    # Dữ liệu demo
├── config/                           # File cấu hình
├── outputs/                          # Kết quả xuất ra
├── Trash/                            # File cũ/không dùng
├── requirements.txt                  # Danh sách thư viện
├── DAP391m_StockMarketAnalysis_CleanReport.md  # Báo cáo chính
└── HUONG_DAN_SU_DUNG.md             # File này
```

---

## GIẢI THÍCH CÁC CHỈ BÁO KỸ THUẬT

### RSI (Relative Strength Index)
- **Giá trị:** 0-100
- **Overbought (Quá mua):** RSI > 70 → Giá có thể giảm
- **Oversold (Quá bán):** RSI < 30 → Giá có thể tăng
- **Trung tính:** RSI 40-60

### MACD (Moving Average Convergence Divergence)
- **Tín hiệu mua:** MACD cắt lên trên đường Signal
- **Tín hiệu bán:** MACD cắt xuống dưới đường Signal
- **Histogram:** Đo độ mạnh của xu hướng

### Bollinger Bands
- **Upper Band:** Ngưỡng trên (kháng cự)
- **Lower Band:** Ngưỡng dưới (hỗ trợ)
- **Giá chạm Upper Band:** Có thể quá mua
- **Giá chạm Lower Band:** Có thể quá bán

### Moving Averages (MA)
- **MA20:** Xu hướng ngắn hạn (20 ngày)
- **MA50:** Xu hướng trung hạn (50 ngày)
- **MA200:** Xu hướng dài hạn (200 ngày)
- **Golden Cross:** MA50 cắt lên MA200 → Tín hiệu tăng mạnh
- **Death Cross:** MA50 cắt xuống MA200 → Tín hiệu giảm mạnh

---

## XỬ LÝ SỰ CỐ THƯỜNG GẶP

### Lỗi: "ModuleNotFoundError"
**Nguyên nhân:** Chưa cài đặt đủ thư viện
**Giải pháp:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Lỗi: "Port 8507 already in use"
**Nguyên nhân:** Port đang được sử dụng
**Giải pháp:**
```bash
pkill -f streamlit
# Hoặc
lsof -ti:8507 | xargs kill -9
```

### Lỗi: "Error fetching stock data"
**Nguyên nhân:** Không có kết nối internet hoặc Yahoo Finance bị lỗi
**Giải pháp:**
1. Kiểm tra kết nối internet
2. Thử lại sau vài phút
3. Sử dụng Demo Mode thay thế

### Chatbot không hoạt động
**Nguyên nhân:** Chưa cấu hình API keys
**Giải pháp:**
1. Tạo file `.env`
2. Thêm API keys của OpenAI và Gemini
3. Khởi động lại ứng dụng

### Ứng dụng chạy chậm
**Giải pháp:**
1. Đóng các tab trình duyệt không dùng
2. Giảm khoảng thời gian phân tích
3. Xóa cache: Settings → Clear Cache

---

## HIỆU SUẤT VÀ GIỚI HẠN

### Hiệu suất
- **Thời gian tải trang:** < 3 giây
- **Thời gian cập nhật biểu đồ:** < 2 giây
- **Thời gian dự đoán AI:** 5-10 giây
- **Thời gian phản hồi chatbot:** 1-3 giây

### Giới hạn
- **Số người dùng đồng thời:** 10-15 users
- **API rate limit:** 2000 requests/day (Yahoo Finance)
- **Dữ liệu lịch sử:** Tối đa 5 năm
- **Mã cổ phiếu:** Chủ yếu thị trường Mỹ (US stocks)

---

## TÍNH NĂNG TƯƠNG LAI

Các tính năng dự kiến phát triển:

### Quý 1/2025
- Hỗ trợ thị trường chứng khoán Việt Nam
- Tích hợp phân tích sentiment từ mạng xã hội
- Mobile app (iOS & Android)

### Quý 2/2025
- Phân tích cryptocurrency
- Tối ưu hóa danh mục đầu tư tự động
- Cảnh báo giá qua email/SMS

### Quý 3/2025
- Hỗ trợ đa ngôn ngữ (Tiếng Việt, Tiếng Anh, Tiếng Trung)
- Quản lý người dùng doanh nghiệp
- Xuất báo cáo PDF/Excel nâng cao

---

## THÔNG TIN LIÊN HỆ VÀ HỖ TRỢ

**Sinh viên thực hiện:** [Tên của bạn]  
**MSSV:** [Mã số sinh viên]  
**Lớp:** DAP391m  
**Email:** vudjeuvuj84@gmail.com  

**Giảng viên hướng dẫn:** [Tên giảng viên]  
**Trường:** FPT University  
**Học kỳ:** Fall 2025  

---

## KẾT LUẬN

Ứng dụng Phân tích Thị trường Chứng khoán là một công cụ mạnh mẽ kết hợp:
- Phân tích kỹ thuật chuyên nghiệp
- Machine Learning và AI tiên tiến
- Giao diện thân thiện, dễ sử dụng
- Tính năng giải thích AI minh bạch

**Mục tiêu:** Hỗ trợ nhà đầu tư cá nhân đưa ra quyết định thông minh, dựa trên dữ liệu và AI.

**Lưu ý quan trọng:** Ứng dụng này phục vụ mục đích học tập và nghiên cứu. Không phải lời khuyên đầu tư tài chính. Người dùng tự chịu trách nhiệm về quyết định đầu tư của mình.

---

**Phiên bản:** 1.0  
**Cập nhật lần cuối:** 03/11/2025  
**Giấy phép:** Educational Use Only

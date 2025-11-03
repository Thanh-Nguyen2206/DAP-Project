# HƯỚNG DẪN PUSH PROJECT LÊN GITHUB

## BƯỚC 1: TẠO REPOSITORY TRÊN GITHUB

1. Truy cập: https://github.com
2. Đăng nhập tài khoản GitHub của bạn
3. Click nút **"New"** hoặc **"+"** ở góc trên bên phải → chọn **"New repository"**
4. Điền thông tin:
   - **Repository name:** `stock-market-analysis` (hoặc tên bạn muốn)
   - **Description:** "AI-powered stock market analysis platform with ML prediction and explainable AI"
   - **Visibility:** Chọn **Public** hoặc **Private**
   - **KHÔNG TICK** vào "Add a README file" (vì chúng ta đã có rồi)
   - Click **"Create repository"**

5. GitHub sẽ hiển thị trang với URL repository, ví dụ:
   ```
   https://github.com/YOUR_USERNAME/stock-market-analysis.git
   ```
   **LƯU LẠI URL NÀY!**

---

## BƯỚC 2: CẤU HÌNH GIT LOCAL

Mở Terminal trong VSCode và chạy các lệnh sau:

### 2.1. Cấu hình thông tin cá nhân (nếu chưa làm)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2.2. Kiểm tra Git đã được khởi tạo
```bash
git status
```

Nếu thấy "Initialized empty Git repository" là OK!

---

## BƯỚC 3: ADD VÀ COMMIT FILES

### 3.1. Add tất cả files vào staging
```bash
git add .
```

### 3.2. Kiểm tra files đã được add
```bash
git status
```

Bạn sẽ thấy danh sách files màu xanh (đã được staged)

### 3.3. Commit với message
```bash
git commit -m "Initial commit: Stock Market Analysis with AI Integration"
```

---

## BƯỚC 4: KẾT NỐI VỚI GITHUB REPOSITORY

### 4.1. Thêm remote origin
**Thay YOUR_USERNAME bằng username GitHub của bạn:**

```bash
git remote add origin https://github.com/YOUR_USERNAME/stock-market-analysis.git
```

### 4.2. Kiểm tra remote đã được thêm
```bash
git remote -v
```

Sẽ hiển thị:
```
origin  https://github.com/YOUR_USERNAME/stock-market-analysis.git (fetch)
origin  https://github.com/YOUR_USERNAME/stock-market-analysis.git (push)
```

---

## BƯỚC 5: PUSH LÊN GITHUB

### 5.1. Đổi tên branch sang main (nếu cần)
```bash
git branch -M main
```

### 5.2. Push lên GitHub
```bash
git push -u origin main
```

**Lưu ý:** Lần đầu push, GitHub sẽ yêu cầu đăng nhập:
- **Username:** Nhập username GitHub của bạn
- **Password:** Nhập **Personal Access Token** (KHÔNG phải password thông thường)

### 5.3. Tạo Personal Access Token (nếu chưa có)
1. GitHub → Click avatar → **Settings**
2. Kéo xuống dưới cùng → **Developer settings**
3. **Personal access tokens** → **Tokens (classic)**
4. **Generate new token** → **Generate new token (classic)**
5. Đặt tên: "Stock Market Analysis Project"
6. Chọn scope: Tick vào **repo** (toàn bộ)
7. Click **Generate token**
8. **COPY TOKEN VÀ LƯU LẠI** (chỉ hiển thị 1 lần!)
9. Dùng token này làm password khi push

---

## BƯỚC 6: XÁC NHẬN ĐÃ PUSH THÀNH CÔNG

1. Truy cập repository trên GitHub:
   ```
   https://github.com/YOUR_USERNAME/stock-market-analysis
   ```

2. Kiểm tra:
   - ✅ Tất cả files và folders đã xuất hiện
   - ✅ README.md hiển thị đẹp
   - ✅ Code có syntax highlighting

3. Repository structure sẽ trông như này:
   ```
   stock-market-analysis/
   ├── .gitignore
   ├── README.md
   ├── requirements.txt
   ├── src/
   ├── docs/
   └── ...
   ```

---

## CÁC LỆNH GIT HỮU ÍCH

### Kiểm tra trạng thái
```bash
git status
```

### Xem lịch sử commit
```bash
git log --oneline
```

### Update code mới
```bash
git add .
git commit -m "Your commit message here"
git push origin main
```

### Pull code mới từ GitHub
```bash
git pull origin main
```

### Xem remote URL
```bash
git remote -v
```

### Thay đổi remote URL
```bash
git remote set-url origin https://github.com/NEW_USERNAME/stock-market-analysis.git
```

---

## XỬ LÝ LỖI THƯỜNG GẶP

### Lỗi: "fatal: not a git repository"
**Giải pháp:**
```bash
cd /Users/vudjeuvuj84gmail.com/Downloads/STUDY/FPTU/2025/DAP391m/Project
git init
```

### Lỗi: "remote origin already exists"
**Giải pháp:**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/stock-market-analysis.git
```

### Lỗi: "failed to push some refs"
**Giải pháp:**
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

### Lỗi: "Authentication failed"
**Giải pháp:**
- Sử dụng Personal Access Token thay vì password
- Hoặc cấu hình SSH key

---

## LỆNH NHANH - COPY VÀ PASTE

**Lệnh đầy đủ để push lần đầu:**

```bash
# Di chuyển vào project
cd /Users/vudjeuvuj84gmail.com/Downloads/STUDY/FPTU/2025/DAP391m/Project

# Khởi tạo git (nếu chưa có)
git init

# Cấu hình user (thay YOUR_NAME và YOUR_EMAIL)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Add tất cả files
git add .

# Commit
git commit -m "Initial commit: Stock Market Analysis with AI Integration"

# Thêm remote (thay YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/stock-market-analysis.git

# Đổi branch sang main
git branch -M main

# Push lên GitHub
git push -u origin main
```

**Lệnh update code sau này:**

```bash
cd /Users/vudjeuvuj84gmail.com/Downloads/STUDY/FPTU/2025/DAP391m/Project
git add .
git commit -m "Update: describe your changes here"
git push origin main
```

---

## CHECKLIST HOÀN THÀNH

- [ ] Đã tạo repository trên GitHub
- [ ] Đã tạo file .gitignore
- [ ] Đã tạo file README.md
- [ ] Đã chạy git init
- [ ] Đã git add và git commit
- [ ] Đã thêm remote origin
- [ ] Đã push thành công
- [ ] Đã kiểm tra trên GitHub
- [ ] README.md hiển thị đẹp
- [ ] Code có syntax highlighting

---

## LỜI KHUYÊN

1. **Không commit API keys:** File .gitignore đã loại trừ .env và API keys
2. **Commit thường xuyên:** Mỗi khi có thay đổi quan trọng
3. **Viết commit message rõ ràng:** Mô tả ngắn gọn những gì đã thay đổi
4. **Sử dụng branches:** Để phát triển tính năng mới
5. **Pull trước khi push:** Nếu làm việc nhóm

---

**Chúc bạn push thành công!** 🚀

Nếu gặp lỗi, hãy copy error message và hỏi tôi để được hỗ trợ.

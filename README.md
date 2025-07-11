# 🎯 Text-Based Object Retrieval in Surveillance Video

Hệ thống này cho phép người dùng **tìm kiếm đối tượng trong video giám sát bằng mô tả văn bản**, ví dụ: `"a red motorcycle"` hoặc `"a green chair"`. Hệ thống sử dụng **YOLO** để nhận diện đối tượng trong video, lưu trữ ảnh đối tượng vào cơ sở dữ liệu, và dùng **CLIP** để truy vấn văn bản gần giống về mặt ngữ nghĩa.

---

## 📌 Mục lục

- [🧠 Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [⚙️ Yêu cầu](#yêu-cầu)
- [🚀 Cách chạy](#cách-chạy)
- [💻 API sử dụng](#api-sử-dụng)
- [🧪 Ví dụ sử dụng](#ví-dụ-sử-dụng)
- [📂 Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [💬 Liên hệ / Đóng góp](#liên-hệ--đóng-góp)

---

## 🧠 Kiến trúc hệ thống

![System Architecture](diagram.png)

**Sơ đồ hệ thống gồm 3 thành phần chính:**
1. **Object Detection**: Trích xuất các đối tượng từ video bằng YOLO.
2. **Feature Indexing**: Sử dụng CLIP để mã hóa hình ảnh và lưu vào FAISS index.
3. **Text Query**: Người dùng nhập mô tả văn bản → CLIP mã hóa → FAISS tìm kiếm ảnh gần nhất.

---

## ⚙️ Yêu cầu

- **Python**: >= 3.8
- **Công cụ**: `pip` + `venv` (hoặc `conda`)
- **Thư viện Python**:

```bash
pip install git+https://github.com/openai/CLIP.git
pip install faiss-cpu flask opencv-python torch torchvision
```

- **Tệp phụ thuộc** (nếu có): `requirements.txt`

---

## 🚀 Cách chạy

1. **Tạo và kích hoạt môi trường ảo**:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
# Trên macOS/Linux:
source venv/bin/activate
# Trên Windows:
.\venv\Scripts\activate
```

2. **Cài đặt các thư viện phụ thuộc**:

```bash
pip install -r requirements.txt
```

3. **Chạy Flask server**:

```bash
python app.py
```

4. **Gửi video để xử lý bằng YOLO**:

```bash
curl -X POST http://localhost:5000/process_video \
     -H "Content-Type: application/json" \
     -d '{"video_path": "data/videos/camera1.mp4"}'
```

5. **Truy vấn tìm kiếm đối tượng bằng văn bản**:

```bash
curl "http://localhost:5000/search?query=a red motorcycle"
```

---

## 💻 API sử dụng

### `POST /process_video`
- **Mô tả**: Trích xuất đối tượng từ video bằng YOLO, lưu metadata, và mã hóa vector bằng CLIP.
- **Request body (JSON)**:

```json
{
  "video_path": "data/videos/camera1.mp4"
}
```

### `GET /search?query=your text`
- **Mô tả**: Tìm các ảnh đối tượng gần nhất với mô tả văn bản.
- **Query parameters**:
  - `query`: Chuỗi mô tả đối tượng (bắt buộc).
  - `k`: Số kết quả trả về (tùy chọn, mặc định: 5).

---

## 🧪 Ví dụ sử dụng

- `"a red motorcycle"`: Trả về ảnh xe máy màu đỏ từ video.
- `"a green chair"`: Trả về ảnh ghế màu xanh nếu có trong video.

---

## 📂 Cấu trúc thư mục

```
object_search_backend/
├── app.py                # Flask backend chính
├── detect.py             # Trích xuất ảnh đối tượng từ video
├── indexer.py            # Mã hóa ảnh bằng CLIP và lưu FAISS index
├── search.py             # Tìm kiếm ảnh gần nhất theo query text
├── data/
│   ├── videos/           # Video đầu vào
│   └── objects/          # Ảnh cắt từ video
├── faiss.index           # Chỉ mục vector ảnh (dạng FAISS)
├── metadata.json         # Thông tin mô tả từng ảnh object
└── diagram.png           # Ảnh sơ đồ kiến trúc hệ thống
```

---

## 💬 Liên hệ / Đóng góp

Bạn có thể mở issue hoặc pull request để:
- Thêm UI hiển thị kết quả.
- Hỗ trợ multi-camera.
- Gợi ý theo thời gian.

---

**Lưu ý**:
1. Lưu nội dung này thành file `README.md` trong thư mục gốc dự án.
2. Đảm bảo ảnh `diagram.png` tồn tại (nếu tên ảnh khác, chỉnh sửa dòng Markdown tương ứng).

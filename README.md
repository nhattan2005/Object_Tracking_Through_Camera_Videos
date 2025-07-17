# Multi-Camera ReID Pipeline - Hướng dẫn chạy dự án

## Mục tiêu:

Xây dựng pipeline cho bài toán **Multi-camera ReID (Person Search)**:

- Dò theo đối tượng (Person Tracking) trên video.
- Trích xuất đặc trưng CLIP.
- Index với FAISS.
- Tìm kiếm đối tượng theo text.

---

## **1. Chuẩn bị môi trường**

### **Cài Python 3.10+**

Tạo môi trường:

```bash
conda create -n mcmot python=3.10
conda activate mcmot
```

---

## **2. Chuẩn bị dữ liệu**

### **Folder dữ liệu video:**

```
D:/MCMOT/data/videos/
    camera3.mp4
    camera4.mp4
```

---

## **3. Cách chạy pipeline**

### **Bước 1: Chạy Flask API**

File API: `app.py`

```bash
python app.py
```

API gồm 2 endpoint:

- **/process\_video\_track**: tracking + extract feature + index.
- **/search\_reid**: tìm kiếm bằng text.

---

### **Bước 2: Tracking và trích xuất đặc trưng**

Gọi request bằng PowerShell hoặc cURL:

#### Camera 3:

```bash
Invoke-RestMethod -Uri http://127.0.0.1:5000/process_video_track `
    -Method POST `
    -Body '{"video_path": "D:/MCMOT/data/videos/camera3.mp4"}' `
    -ContentType "application/json"
```

#### Camera 4:

```bash
Invoke-RestMethod -Uri http://127.0.0.1:5000/process_video_track `
    -Method POST `
    -Body '{"video_path": "D:/MCMOT/data/videos/camera4.mp4"}' `
    -ContentType "application/json"
```

Kết quả:

- 2 video sẽ được tracking.
- Đặc trưng CLIP sẽ được index vào **reid.index**.
- File metadata lưu tại **reid\_metadata.json**.

---

### **Bước 3: Tìm kiếm ReID theo text**

Ví dụ:

#### Tìm người mặc áo xanh:

```bash
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search_reid?query=a person with green shirt&k=5"
```

#### Tìm người mặc áo cam:

```bash
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search_reid?query=a child with orange shirt&k=5"
```

Kết quả trả về:

- Danh sách bounding box, frame, path ảnh crop.
- Video tương ứng.


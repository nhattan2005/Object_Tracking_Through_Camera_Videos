# Multi-Camera ReID Pipeline

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
conda create -n mcmot3 python=3.10
conda activate mcmot3
```

### **Cài đặt thư viện:**

```bash
pip install -r requirements.txt
```

---

## **2. Clone và cài đặt ByteTrack**

Pipeline sử dụng **BYTETracker** từ ByteTrack để thực hiện Multi-Object Tracking.

### **Bước 1: Clone ByteTrack**

```bash
git clone https://github.com/ifzhang/ByteTrack.git
```

### **Bước 2: Cài đặt ByteTrack**

```bash
cd ByteTrack
pip install -r requirements.txt
```

---

## **3. Chuẩn bị dữ liệu**

### **Folder dữ liệu video:**

```
MCMOT/data/videos/
    camera3.mp4
    camera4.mp4
```

---

## **4. Cách chạy pipeline**

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

---

## **5. Cấu trúc file chính**

| File              | Chức năng                       |
| ----------------- | ------------------------------- |
| `app.py`          | API Flask                       |
| `my_tracker.py`   | Tracking + extract CLIP feature |
| `reid_indexer.py` | FAISS index                     |
| `search_reid.py`  | Tìm kiếm bằng text              |



# Project_Traffic_Sign_Detection

## Object Detection Pipeline – 2 Phases

Bài toán nhận diện biển báo giao thông được chia thành hai pha chính:

### Phase 1: Classification (Phân loại đối tượng)

- `Mục tiêu`: huấn luyện một mô hình phân loại có khả năng nhận dạng các loại biển báo giao thông từ ảnh đã được cắt sẵn.
- Các bước thực hiện:
  - Trích xuất ảnh đối tượng từ file annotation (dựa trên bounding boxes).

  - Tiền xử lý ảnh: chuyển grayscale, resize về 32×32.

  - Trích xuất đặc trưng HOG (Histogram of Oriented Gradients).

  - Mã hóa nhãn bằng LabelEncoder.

  - Huấn luyện mô hình phân loại SVM với kernel RBF.

  - Đánh giá độ chính xác trên tập validation và test.

`Output` của Phase 1 là một mô hình phân loại SVM có khả năng phân biệt các loại biển báo giao thông.

### Phase 2: Localization & Evaluation (Xác định vị trí & đánh giá)
- `Mục tiêu`: tìm vị trí xuất hiện của biển báo trong ảnh gốc và đánh giá độ chính xác của hệ thống.
- Các bước thực hiện:

  - Áp dụng image pyramid để tạo ảnh đa tỉ lệ.

  - Duyệt ảnh bằng sliding window với các kích thước cửa sổ khác nhau.

  - Với mỗi cửa sổ: trích xuất đặc trưng HOG và phân loại bằng SVM đã huấn luyện.

  - Giữ lại các cửa sổ có xác suất phân loại cao (trên ngưỡng).

  - Áp dụng NMS (Non-Maximum Suppression) để loại bỏ các bounding box chồng lắp.

  - So sánh với ground-truth bằng chỉ số IoU và đánh giá AP (Average Precision) cho từng lớp, tính mAP (mean AP) toàn bộ.

- `Output` Kết quả được trực quan hóa bằng hình ảnh và lưu vào thư mục output_test.
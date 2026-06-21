# 📊 Data Visualization & Machine Learning Notebooks

Chào mừng bạn đến với kho lưu trữ các dự án Trực quan hóa dữ liệu (Data Visualization) và Học máy (Machine Learning). Đây là tập hợp các nghiên cứu chuyên sâu về phân tích xu hướng, khám phá dữ liệu (EDA) và xây dựng mô hình dự báo trên nhiều lĩnh vực thực tiễn khác nhau như điện ảnh, kinh tế, khí tượng học và giáo dục.

Dự án được thực hiện bởi nhóm sinh viên lớp Trực quan hóa dữ liệu (Chiều Thứ 7):
*   **Võ Bạch Kim Thịnh** (MSSV: 2045230096)
*   **Phan Trọng Nguyên** (MSSV: 2045230068)
*   **Lê Kim Ngân** (MSSV: 2045230067)

---

## 📌 Bảng Tổng Quan Dự Án (Project Summary)

Dưới đây là tóm tắt nhanh về 5 notebook chính trong kho lưu trữ này:

| Tên File Notebook | Lĩnh Vực / Chủ Đề | Thư Viện Trực Quan | Mô Hình Machine Learning | Biến Mục Tiêu (Target) |
| :--- | :--- | :--- | :--- | :--- |
| [1980_2026_Top_Movies.ipynb](notebooks/1980_2026_Top_Movies.ipynb) | Điện ảnh (IMDb Top Movies) | Matplotlib, Seaborn | Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost, Stacking Regressor | `average_rating` (Hồi quy) & Phân loại Hit/Flop |
| [Housing_Affordability_Analysis_and_Prediction.ipynb](notebooks/Housing_Affordability_Analysis_and_Prediction.ipynb) | Kinh tế vĩ mô (Housing Affordability) | Matplotlib, Seaborn | Ridge, Random Forest, HistGradientBoostingRegressor | Chỉ số khả năng chi trả nhà ở (Housing Affordability Index) |
| [Rainfall_Prediction.ipynb](notebooks/Rainfall_Prediction.ipynb) | Khí tượng học (Rainfall in UP, India) | Matplotlib, Seaborn | Random Forest, XGBoost, LightGBM | `PRECTOTCORR` (Lượng mưa) & Cảnh báo mưa nhị phân |
| [Student_Lifestyle_and_Stress_Prediction.ipynb](notebooks/Student_Lifestyle_and_Stress_Prediction.ipynb) | Giáo dục & Tâm lý học (Student Lifestyle) | Matplotlib, Seaborn | Logistic Regression, Random Forest, XGBoost, Stacking Classifier | Mức độ căng thẳng (Stress Level) |
| [Plotly_Basics1.ipynb](notebooks/Plotly_Basics1.ipynb) | Hướng dẫn Trực quan hóa (Plotly Guide) | Plotly Express, Plotly Graph Objects | *N/A (Chỉ trực quan hóa dữ liệu)* | *N/A* |

---

## 🔍 Chi Tiết Từng Dự Án (Detailed Project Description)

### 1. 🎬 Phân Tích & Dự Đoán Phim Top IMDb (1980–2026)
*   **Mục tiêu**: Phân tích các yếu tố quyết định sự thành công của một tác phẩm điện ảnh (thời lượng, thể loại, số lượt bình chọn) và dự đoán điểm đánh giá trung bình.
*   **Tiền xử lý & Kỹ thuật đặc trưng**:
    *   Mã hóa đa nhãn (Multi-Label One-Hot Encoding) cho cột thể loại phim (`genres`).
    *   Tính toán độ tuổi của phim dựa trên năm sản xuất.
    *   Chuẩn hóa dữ liệu bằng `StandardScaler`.
*   **Mô hình sử dụng**: So sánh hiệu năng giữa Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost và Stacking Regressor. Đánh giá chéo qua K-Fold Cross-Validation và kiểm soát overfitting.

### 2. 🏠 Phân Tích Khả Năng Chi Trả Nhà Ở Toàn Cầu (1975–2025)
*   **Mục tiêu**: Nghiên cứu xu hướng biến động của thị trường bất động sản tại 26 quốc gia trong 50 năm qua và dự đoán khả năng tiếp cận nhà ở của người dân.
*   **Điểm nổi bật**:
    *   **Feature Engineering chuyên sâu**: Tạo các biến trễ (Lag Features 1 năm, 2 năm) cho chỉ số giá nhà và thu nhập khả dụng; tính toán chỉ số trung bình động (Rolling Averages 3 năm) để làm mịn xu hướng thời gian.
    *   **Phân chia dữ liệu theo dòng thời gian (Temporal Splitting)**: Dùng dữ liệu trước năm 2018 để huấn luyện và dữ liệu từ 2019-2025 để kiểm thử nhằm đảm bảo tính thực tế của mô hình dự báo chuỗi thời gian.
*   **Mô hình**: Ridge, Random Forest Regressor và HistGradientBoostingRegressor.

### 3. 🌧️ Dự Báo Lượng Mưa tại Uttar Pradesh, Ấn Độ (2005–2025)
*   **Mục tiêu**: Dự đoán chính xác lượng mưa hàng ngày và đưa ra cảnh báo sớm về các hiện tượng mưa lớn gây ngập lụt.
*   **Quy trình thực hiện**:
    *   EDA trực quan hóa lượng mưa theo tháng và năm để nhận diện tính chu kỳ theo mùa.
    *   Tạo các đặc trưng thời gian (Season, Month, Quarter) kết hợp Target Encoding cho thông tin địa lý (các quận huyện tại bang Uttar Pradesh).
    *   Xây dựng song song hai bài toán: **Hồi quy** (dự báo lượng mưa chi tiết bằng `XGBRegressor`, `LGBMRegressor`) và **Phân loại** (cảnh báo mưa/không mưa bằng mô hình phân loại nhị phân).

### 4. 🧠 Dự Đoán Mức Độ Căng Thẳng Ở Học Sinh/Sinh Viên
*   **Mục tiêu**: Nghiên cứu sự ảnh hưởng của thói quen sinh hoạt (thời gian học tập, ngủ nghỉ, hoạt động thể chất, thời gian sử dụng thiết bị điện tử) tới mức độ căng thẳng của học sinh.
*   **Kỹ thuật xử lý**:
    *   Sử dụng `KNNImputer` để xử lý dữ liệu khuyết thiếu một cách thông minh dựa trên các đặc tính sinh hoạt tương tự.
    *   Áp dụng kỹ thuật cân bằng dữ liệu `SMOTE` (Synthetic Minority Over-sampling Technique) giải quyết vấn đề mất cân bằng giữa các lớp stress.
    *   Mô hình Stacked Classifier kết hợp Logistic Regression, Random Forest và XGBoost để tối ưu hóa chỉ số F1-Score và ROC-AUC.

### 5. 📊 Cẩm Nang Trực Quan Hóa Tương Tác Với Plotly
*   **Mục tiêu**: Tài liệu hướng dẫn sử dụng thư viện Plotly để xây dựng biểu đồ tương tác cao.
*   **Các biểu đồ minh họa**: Scatter Plot (Biểu đồ phân tán), Line Plot (Biểu đồ đường), Bar Plot (Biểu đồ cột), Histogram (Biểu đồ phân phối), Bubble Plot (Biểu đồ bong bóng), Pie Plot (Biểu đồ tròn), và Sunburst Charts (Biểu đồ hướng dương phân tầng).

---

## 🛠️ Công Nghệ & Thư Viện Sử Dụng (Tech Stack)

Dự án được xây dựng trên ngôn ngữ **Python 3** và sử dụng các thư viện cốt lõi sau:

*   **Xử lý dữ liệu**: `pandas`, `numpy`
*   **Trực quan hóa**: `matplotlib`, `seaborn`, `plotly`
*   **Học máy & Tiền xử lý**: `scikit-learn` (StandardScaler, RobustScaler, KNNImputer, StackingClassifier)
*   **Thuật toán Boosting mạnh mẽ**: `xgboost`, `lightgbm`, `imbalanced-learn` (SMOTE)

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Dự Án (Installation & Setup)

### Chạy tại máy cục bộ (Local Setup)

1. **Clone kho lưu trữ về máy**:
   ```bash
   git clone git@github.com:KimThinh212/Visualization.git
   cd Visualization
   ```

2. **Tạo môi trường ảo (Virtual Environment) và kích hoạt**:
   - Trên Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Trên macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Cài đặt các thư viện cần thiết**:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm imbalanced-learn notebook
   ```

4. **Khởi chạy Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

### Chạy trên Google Colab
Bạn có thể upload các file notebook này trực tiếp lên Google Drive và mở bằng Google Colab. Đảm bảo cấu hình đường dẫn dữ liệu hoặc tải dữ liệu lên thư mục `/content/` của phiên làm việc Colab.

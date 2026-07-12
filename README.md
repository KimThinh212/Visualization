# 📊 Data Visualization & Machine Learning Notebooks

Chào mừng bạn đến với kho lưu trữ các dự án Trực quan hóa dữ liệu (Data Visualization) kết hợp Học máy (Machine Learning) và Học sâu (Deep Learning). Đây là tập hợp các nghiên cứu chuyên sâu về phân tích xu hướng, khám phá dữ liệu (EDA), kiểm định giả thuyết thống kê và xây dựng mô hình dự báo trên nhiều lĩnh vực thực tiễn như điện ảnh, tài chính công, kinh tế vĩ mô, khí tượng học và giáo dục.

Dự án được thực hiện bởi:
*   **Võ Bạch Kim Thịnh** (MSSV: 2045230096)

---

## 📌 Bảng Tổng Quan Dự Án (Project Summary)

Dưới đây là danh mục 6 notebook chính trong kho lưu trữ này:

| Tên File Notebook | Lĩnh Vực / Chủ Đề | Thư Viện Trực Quan | Kỹ Thuật & Mô Hình Machine/Deep Learning | Biến Mục Tiêu (Target) |
| :--- | :--- | :--- | :--- | :--- |
| [1980_2026_Top_Movies.ipynb](notebooks/1980_2026_Top_Movies.ipynb) | Điện ảnh (IMDb Top Movies) <br> [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/esrakoyun/1980-2026-top-movies) | Matplotlib, Seaborn | Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost, Stacking Regressor | `average_rating` (Hồi quy) & Phân loại Hit/Flop |
| [Global_Budget_Allocation_Dataset_(1936–2026).ipynb](notebooks/Global_Budget_Allocation_Dataset_%281936%E2%80%932026%29.ipynb) | Tài chính công (Global Budget) <br> [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/ashyou09/global-budget-allocation-dataset-19362026) | Matplotlib, Seaborn, Plotly | **Học máy**: Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM. <br>**Học sâu**: Stacked LSTM (Keras/TF), Kiểm định K-S, Levene, T-test, ADF. | `Total_Budget_Billions_USD` (Dự báo T+1), `Healthcare_Budget_Billions_USD` & Chuỗi thời gian VN |
| [Housing_Affordability_Analysis_and_Prediction.ipynb](notebooks/Housing_Affordability_Analysis_and_Prediction.ipynb) | Kinh tế vĩ mô (Housing Affordability) <br> [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/lucalullo/housing-affordability-26-countries-1975-2025) | Matplotlib, Seaborn | Ridge, Random Forest, HistGradientBoostingRegressor, Lag Features, Rolling Window | Chỉ số khả năng chi trả nhà ở (Housing Affordability Index) |
| [Rainfall_Prediction.ipynb](notebooks/Rainfall_Prediction.ipynb) | Khí tượng học (Rainfall in UP, India) <br> [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/rupsarroy/rainfall-dataset-uttar-pradesh-20052025) | Matplotlib, Seaborn | Random Forest, XGBoost, LightGBM, Target Encoding, Phân loại & Hồi quy song song | `PRECTOTCORR` (Lượng mưa) & Cảnh báo mưa nhị phân |
| [Student_Lifestyle_and_Stress_Prediction.ipynb](notebooks/Student_Lifestyle_and_Stress_Prediction.ipynb) | Giáo dục & Tâm lý (Student Lifestyle) <br> [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/rxnach/student-lifestyle-and-stress-dataset) | Matplotlib, Seaborn | KNNImputer, Ordinal Encoding, SMOTE, Stacking Classifier (Logistic Regression, RF, XGBoost) | Mức độ căng thẳng (Stress Level) |
| [TMDB_Top_10,000_Movies.ipynb](notebooks/TMDB_Top_10,000_Movies.ipynb) | Điện ảnh (TMDB Top 10k Movies) <br> [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/siddharthbhakta/tmdb-top-10000-movies-updated-2026) | Matplotlib, Seaborn | Linear Regression, Ridge, Random Forest, Gradient Boosting, One-way ANOVA, IQR Outlier Detection | `vote_average` (Dự đoán điểm phim) |

---

## 🔍 Chi Tiết Từng Dự Án (Detailed Project Description)

### 1. 🎬 Phân Tích & Dự Đoán Phim Top IMDb (1980–2026)
*   **Mục tiêu**: Phân tích các yếu tố quyết định sự thành open của một tác phẩm điện ảnh (thời lượng, thể loại, số lượt bình chọn) và dự đoán điểm đánh giá trung bình.
*   **Nguồn dữ liệu (Kaggle)**: [IMDb Top Movies 1980-2026 Dataset](https://www.kaggle.com/datasets/esrakoyun/1980-2026-top-movies)
*   **Tiền xử lý & Kỹ thuật đặc trưng**:
    *   Mã hóa đa nhãn (Multi-Label One-Hot Encoding) cho cột thể loại phim (`genres`).
    *   Tính toán độ tuổi của phim dựa trên năm sản xuất.
    *   Chuẩn hóa dữ liệu bằng `StandardScaler`.
*   **Mô hình sử dụng**: Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost và Stacking Regressor. Đánh giá chéo qua K-Fold Cross-Validation và kiểm soát overfitting.

### 2. 🏛️ Phân Tích & Dự Báo Ngân Sách Chính Phủ Toàn Cầu (1936–2026)
*   **Mục tiêu**: Phân tích toàn diện cơ cấu phân bổ tài chính công toàn cầu, kiểm định giả thuyết thống kê về phân bổ ngân sách và dự báo xu hướng ngân sách tương lai (T+1) và chuỗi thời gian quốc gia.
*   **Nguồn dữ liệu (Kaggle)**: [Global Government Budget Dataset 1936–2026](https://www.kaggle.com/datasets/ashyou09/global-budget-allocation-dataset-19362026)
*   **Tiến trình thực hiện**:
    *   **Tải và Nạp dữ liệu tự động**: Tích hợp Kaggle API thông qua thư viện `kagglehub` để tự động kéo phiên bản dataset mới nhất.
    *   **Kiểm định thống kê chuyên sâu**: Thực hiện kiểm định Kolmogorov-Smirnov (kiểm tra tính chuẩn), Levene Test (kiểm tra đồng nhất phương sai), và Independent T-test phục vụ cho kiểm thử giả thuyết ngân sách quốc phòng/an sinh.
    *   **Trực quan hóa nâng cao**: Thiết kế hệ thống 11 biểu đồ chuyên sâu, trong đó có Bản đồ phân phối quy mô ngân sách toàn cầu (Global Choropleth Map) thông qua Plotly Express.
    *   **Học máy dự báo tài chính**:
        *   *Bài toán 1*: Dự báo tổng ngân sách năm kế tiếp (T+1) sử dụng Ridge, Random Forest, Gradient Boosting, XGBoost và LightGBM.
        *   *Bài toán 2*: Dự đoán phân bổ ngân sách Y tế dựa trên cơ cấu vĩ mô.
    *   **Học sâu chuỗi thời gian (Việt Nam)**: Thực hiện kiểm định tính dừng Augmented Dickey-Fuller (ADF Test) cho chuỗi thời gian của Việt Nam. Xây dựng kiến trúc **Stacked LSTM** (2 tầng LSTM 64 units xếp chồng, kết hợp Dropout 20%, bộ tối ưu Adam với Learning Rate động qua `ReduceLROnPlateau` và cơ chế dừng sớm `EarlyStopping`).

### 3. 🏠 Phân Tích Khả Năng Chi Trả Nhà Ở Toàn Cầu (1975–2025)
*   **Mục tiêu**: Nghiên cứu xu hướng biến động của thị trường bất động sản tại 26 quốc gia trong 50 năm qua và dự đoán khả năng tiếp cận nhà ở của người dân.
*   **Nguồn dữ liệu (Kaggle)**: [Housing Affordability in 26 countries, 1975-2025](https://www.kaggle.com/datasets/lucalullo/housing-affordability-26-countries-1975-2025)
*   **Điểm nổi bật**:
    *   **Feature Engineering chuyên sâu**: Tạo các biến trễ (Lag Features 1 năm, 2 năm) cho chỉ số giá nhà và thu nhập khả dụng; tính toán chỉ số trung bình động (Rolling Averages 3 năm) để làm mịn xu hướng thời gian.
    *   **Phân chia dữ liệu theo dòng thời gian (Temporal Splitting)**: Dùng dữ liệu trước năm 2018 để huấn luyện và dữ liệu từ 2019-2025 để kiểm thử nhằm đảm bảo tính thực tế của mô hình dự báo chuỗi thời gian.
*   **Mô hình**: Ridge, Random Forest Regressor và HistGradientBoostingRegressor.

### 4. 🌧️ Dự Báo Lượng Mưa tại Uttar Pradesh, Ấn Độ (2005–2025)
*   **Mục tiêu**: Dự đoán chính xác lượng mưa hàng ngày và đưa ra cảnh báo sớm về các hiện tượng mưa lớn gây ngập lụt.
*   **Nguồn dữ liệu (Kaggle)**: [Rainfall Dataset Uttar Pradesh 2005-2025](https://www.kaggle.com/datasets/rupsarroy/rainfall-dataset-uttar-pradesh-20052025)
*   **Quy trình thực hiện**:
    *   EDA trực quan hóa lượng mưa theo tháng và năm để nhận diện tính chu kỳ theo mùa.
    *   Tạo các đặc trưng thời gian (Season, Month, Quarter) kết hợp Target Encoding cho thông tin địa lý (các quận huyện tại bang Uttar Pradesh).
    *   Xây dựng song song hai bài toán: **Hồi quy** (dự báo lượng mưa chi tiết bằng `XGBRegressor`, `LGBMRegressor`) và **Phân loại** (cảnh báo mưa/không mưa bằng mô hình phân loại nhị phân).

### 5. 🧠 Dự Đoán Mức Độ Căng Thẳng Ở Học Sinh/Sinh Viên
*   **Mục tiêu**: Nghiên cứu sự ảnh hưởng của thói quen sinh hoạt (thời gian học tập, ngủ nghỉ, hoạt động thể chất, thời gian sử dụng thiết bị điện tử) tới mức độ căng thẳng của học sinh.
*   **Nguồn dữ liệu (Kaggle)**: [Student Lifestyle and Stress Prediction Dataset](https://www.kaggle.com/datasets/rxnach/student-lifestyle-and-stress-dataset)
*   **Kỹ thuật xử lý**:
    *   Sử dụng `KNNImputer` để xử lý dữ liệu khuyết thiếu một cách thông minh dựa trên các đặc tính sinh hoạt tương tự.
    *   Áp dụng kỹ thuật cân bằng dữ liệu `SMOTE` (Synthetic Minority Over-sampling Technique) giải quyết vấn đề mất cân bằng giữa các lớp stress.
    *   Mô hình Stacked Classifier kết hợp Logistic Regression, Random Forest và XGBoost để tối ưu hóa chỉ số F1-Score và ROC-AUC.

### 6. 🎬 Phân Tích & Dự Đoán Điểm Phim TMDB Top 10,000 (Cập nhật 2026)
*   **Mục tiêu**: Khai phá kho dữ liệu điện ảnh TMDB với hơn 10,000 bản ghi nhằm tìm kiếm xu hướng thị hiếu khán giả và dự đoán điểm đánh giá trung bình của bộ phim.
*   **Nguồn dữ liệu (Kaggle)**: [TMDB Top 10,000 Movies - Updated 2026](https://www.kaggle.com/datasets/siddharthbhakta/tmdb-top-10000-movies-updated-2026)
*   **Tiến trình thực hiện**:
    *   **Làm sạch & Tiền xử lý**: Khảo sát dữ liệu khuyết thiếu, xử lý giá trị NaN cho trường text (`overview`) và ngày phát hành (`release_date`). Loại bỏ các thuộc tính định danh không cần thiết.
    *   **Kỹ nghệ đặc trưng (Feature Engineering)**: Trích xuất năm phát hành (`release_year`), tháng phát hành (`release_month`). Gom nhóm các ngôn ngữ gốc ít phổ biến thành cụm "Others" để tăng hiệu quả mô hình hóa.
    *   **Phân tích Thống kê & Kiểm định ANOVA**: 
        *   Sử dụng phương pháp Khoảng tứ phân vị (IQR) để phát hiện và định lượng các giá trị ngoại lai cho các trường thuộc tính số học (`vote_count`, `popularity`, `revenue`, `budget`).
        *   Thực hiện phân tích tương quan tuyến tính (Pearson) và phi tuyến (Spearman).
        *   Tiến hành kiểm định giả thuyết **One-way ANOVA** độc lập nhằm kiểm nghiệm sự ảnh hưởng của ngôn ngữ gốc đến điểm đánh giá chất lượng phim trung bình (`vote_average`).
    *   **Mô hình hóa**: Chia tập dữ liệu, chuẩn hóa đặc trưng số bằng `StandardScaler`, mã hóa One-Hot cho dữ liệu phân loại. So sánh hiệu năng của 4 mô hình Hồi quy: Linear Regression, Ridge, Random Forest và Gradient Boosting.

---

## 🛠️ Công Nghệ & Thư Viện Sử Dụng (Tech Stack)

Dự án sử dụng ngôn ngữ **Python 3** và các thư viện chuyên sâu sau:

*   **Xử lý dữ liệu**: `pandas`, `numpy`
*   **Trực quan hóa dữ liệu**: `matplotlib`, `seaborn`, `plotly`
*   **Kiểm định thống kê**: `scipy.stats`, `statsmodels`
*   **Học máy & Tiền xử lý**: `scikit-learn` (StandardScaler, RobustScaler, KNNImputer, StackingClassifier, TargetEncoder, ColumnTransformer, Pipeline), `imbalanced-learn` (SMOTE)
*   **Thuật toán Boosting mạnh mẽ**: `xgboost`, `lightgbm`
*   **Học sâu (Deep Learning)**: `tensorflow`, `keras` (kiến trúc Stacked LSTM)
*   **Quản lý & Tải dữ liệu**: `kagglehub`, `glob`, `os`

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
   pip install pandas numpy matplotlib seaborn plotly scipy statsmodels scikit-learn xgboost lightgbm imbalanced-learn tensorflow notebook kagglehub
   ```

4. **Khởi chạy Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

### Chạy trên Google Colab
Bạn có thể upload các file notebook này trực tiếp lên Google Drive và mở bằng Google Colab. Đảm bảo cài đặt các thư viện hỗ trợ bằng cách chạy `!pip install <tên_thư_viện>` ở ô lệnh đầu tiên nếu Colab chưa hỗ trợ sẵn.

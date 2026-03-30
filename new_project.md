
# new project




---

## 1. 방법론 및 다중공선성 처리 (Methodology and Multicollinearity Handling)

### 표 1: VIF 결과 (Table 1: VIF)

VIF 결과에 따르면, **SIZE** 변수는 **12.0922**의 값을 보여 다중공선성(multicollinearity) 수준이 유의미함을 반영하며, 변수들이 많은 정보를 공유할 때 전통적인 선형 회귀 모형이 **왜곡된 계수**를 산출할 수 있음을 경고한다. 이러한 한계로 인해, 비선형 관계 및 변수 간 상호작용을 잘 처리할 수 있으며 비다중공선성 가정을 따르지 않는 기계 학습 모형으로 분석을 확장할 필요성이 제기된다. 따라서 ML로의 전환은 예측 정확도 향상뿐만 아니라, 변수 간 높은 상관관계가 있는 데이터 맥락에서 결과의 견고성(robustness)을 테스트하기 위함이다.

또한, 본 연구는 계량경제학적 모형에서 잔차 변환(residualized transformation)을 적용하여 **SIZE\_RESID** 변수를 생성하였는데, 이는 다중공선성을 최소화하면서 변수의 경제적 유의성을 보존하기 위함이다. SIZE\_RESID 변수를 사용한 VIF 값은 1.0009로 현저히 감소하여, 다중공선성 문제가 성공적으로 해결되었음을 나타낸다.

---

## 2. 모형 성과 및 비교 (Model Performance and Comparison)


## timeseriesCV 
|            |   Train_R2 |   Train_RMSE |   Train_MAE |   Test_R2 |   Test_RMSE |   Test_MAE |
|:-----------|-----------:|-------------:|------------:|----------:|------------:|-----------:|
| Linear | | | | | 
| OLS        |     0.1680 |       1.3067 |      0.6496 |    0.0413 |      1.4989 |     0.8012 |
| Ridge      |     0.1679 |       1.3067 |      0.6494 |    0.0433 |      1.4972 |     0.8004 |
| LASSO      |     0.1676 |       1.3070 |      0.6496 |    0.0509 |      1.4914 |     0.7990 |
| ElasticNet |     0.1678 |       1.3069 |      0.6492 |    0.0465 |      1.4948 |     0.7993 |
|Nonlinear|||||||
| RandomForest |     0.7119 |       0.7690 |      0.3236 |    0.2989 |      1.2817 |     0.7220 |
| XGBoost      |     0.8089 |       0.6263 |      0.3713 |    0.2519 |      1.3240 |     0.7324 |
| LightGBM     |     0.6876 |       0.8007 |      0.4409 |    0.2998 |      1.2809 |     0.7003 |



## rolling 
|              |   Train_R2 |   Train_RMSE |   Train_MAE |   Test_R2 |   Test_RMSE |   Test_MAE |   CV_MSE |
|:-------------|-----------:|-------------:|------------:|----------:|------------:|-----------:|---------:|
| OLS          |     0.1680 |       1.3067 |      0.6496 |    0.0413 |      1.4989 |     0.8012 |   2.4769 |
| Ridge        |     0.1679 |       1.3067 |      0.6494 |    0.0433 |      1.4972 |     0.8004 |   2.4723 |
| LASSO        |     0.1676 |       1.3070 |      0.6496 |    0.0509 |      1.4914 |     0.7990 |   2.4711 |
| ElasticNet   |     0.1632 |       1.3104 |      0.6488 |    0.0624 |      1.4822 |     0.7919 |   2.4711 |
| RandomForest |     0.6686 |       0.8246 |      0.3833 |    0.3002 |      1.2805 |     0.7183 |   2.9785 |
| XGBoost      |     0.6356 |       0.8648 |      0.4828 |    0.3105 |      1.2711 |     0.7087 |   2.1955 |
| LightGBM     |     0.6140 |       0.8900 |      0.4550 |    0.3033 |      1.2777 |     0.6974 |   2.1530 |

#### 
Bảng X trình bày hiệu suất dự đoán ngoài mẫu của các mô hình tuyến tính và phi tuyến tính theo hai phương pháp đánh giá: kiểm định chéo chuỗi thời gian (TSCV) và cửa sổ mở rộng trượt.

Nhìn chung, kết quả nhất quán cho thấy các mô hình học máy phi tuyến tính vượt trội hơn đáng kể so với các mô hình tuyến tính truyền thống trong việc dự đoán giá trị doanh nghiệp. Trên cả hai thiết kế đánh giá, các mô hình tuyến tính—bao gồm OLS, Ridge, LASSO và Elastic Net—cho thấy khả năng giải thích tương đối thấp, với sai số ngoài mẫu nằm trong khoảng từ 0,04 đến 0,06. Điều này cho thấy rằng chỉ riêng các mối quan hệ tuyến tính là không đủ để nắm bắt được quá trình tạo ra dữ liệu về giá trị doanh nghiệp , vốn có thể được đặc trưng bởi các tương tác phức tạp và tính phi tuyến tính.

Ngược lại, các mô hình phi tuyến tính thể hiện hiệu suất vượt trội hơn hẳn. Dưới hệ thống TSCV, Random Forest và LightGBM đạt được khả năng dự đoán ngoài mẫu gần bằng 0,30, trong khi XGBoost đạt giá trị thấp hơn một chút là 0,25. Trong khuôn khổ cửa sổ trượt, hiệu suất tiếp tục được cải thiện đối với một số mô hình nhất định, với XGBoost đạt giá trị ngoài mẫu cao nhất là 0,3105, tiếp theo sát là LightGBM và Random Forest. Những phát hiện này cung cấp bằng chứng thực nghiệm mạnh mẽ ủng hộ việc sử dụng các kỹ thuật học máy phi tuyến tính trong dự đoán giá trị doanh nghiệp , phù hợp với quan điểm cho rằng các đặc điểm của doanh nghiệp tương tác theo cách rất phi tuyến tính.

#### độ quá khớp 
So sánh hiệu năng trong mẫu và ngoài mẫu cho thấy sự khác biệt đáng kể về khả năng khái quát hóa của mô hình. Dưới TSCV, các mô hình phi tuyến tính thể hiện khoảng cách đáng kể giữa hiệu năng huấn luyện và kiểm thử. Ví dụ, XGBoost đạt được hiệu năng trong mẫu là 0,8089 nhưng chỉ 0,2519 khi kiểm tra ngoài mẫu, cho thấy mức độ quá khớp dữ liệu khá rõ rệt. Thuật toán Random Forest cũng cho thấy mô hình tương tự, mặc dù ít nghiêm trọng hơn.

Ngược lại, LightGBM thể hiện sự cân bằng tốt hơn giữa độ phù hợp và khả năng khái quát hóa, với khoảng cách tương đối nhỏ hơn giữa hiệu suất huấn luyện và kiểm thử. Điều này cho thấy LightGBM ít bị quá khớp hơn so với các phương pháp tập hợp dựa trên cây khác , có thể là do cơ chế điều chỉnh và chiến lược phát triển theo từng lá của nó.

Điều quan trọng là, thiết kế cửa sổ mở rộng theo kiểu cuốn chiếu giúp giảm thiểu hiện tượng quá khớp (overfitting) trên tất cả các mô hình phi tuyến tính. Việc giảm khoảng cách giữa tập huấn luyện và tập kiểm tra cho thấy rằng đánh giá theo kiểu cuốn chiếu cung cấp một đánh giá thực tế hơn về hiệu suất dự đoán trong môi trường phụ thuộc thời gian , vì các mô hình được ước tính lại nhiều lần bằng cách sử dụng thông tin có sẵn cho đến mỗi thời điểm.

#### 4.3 Tính ổn định của mô hình và độ nhạy cảm đối với thiết kế đánh giá

Một phát hiện quan trọng của nghiên cứu này là hiệu suất và thứ hạng của mô hình rất nhạy cảm với khung đánh giá . Trong khi Random Forest và LightGBM chiếm ưu thế dưới phương pháp TSCV, XGBoost lại nổi lên là mô hình hoạt động tốt nhất dưới phương pháp cửa sổ trượt.

Sự thay đổi trong xếp hạng mô hình này nhấn mạnh rằng thiết kế đánh giá đóng vai trò quan trọng trong việc xác định tính ưu việt của mô hình . TSCV có xu hướng phạt các mô hình bị quá khớp trong các tập huấn luyện, trong khi đánh giá bằng cửa sổ trượt phản ánh tốt hơn các điều kiện dự báo thực tế bằng cách mô phỏng sự xuất hiện thông tin tuần tự. Kết quả là, các mô hình như XGBoost, ban đầu có vẻ kém cạnh tranh hơn theo TSCV, lại thể hiện hiệu suất mạnh mẽ hơn khi được đánh giá trong một thiết lập thời gian thực tế hơn.

Bất chấp những khác biệt này, LightGBM thể hiện hiệu suất ổn định nhất trên cả hai khung đánh giá, với độ ổn định ngoài mẫu sai số dự đoán tương đối thấp. Tính ổn định này còn được củng cố bởi sai số bình phương trung bình được kiểm định chéo (CV_MSE) thấp nhất trong số tất cả các mô hình phi tuyến tính trong khuôn khổ rolling, và Test MAE của lgbm cũng luôn thấp nhất trong tất cả mô hình sù sử dụng khung đánh giá nào. 


## Mô hình lai ghép theo các thiết kế đánh giá thay thế
Để nghiên cứu sâu hơn liệu việc kết hợp các mô hình tuyến tính và phi tuyến tính có thể nâng cao hiệu suất dự đoán hay không, chúng tôi mở rộng phân tích bằng cách xây dựng các mô hình lai trong mỗi khung đánh giá. Thay vì dựa vào một cặp mô hình cố định, nghiên cứu này áp dụng chiến lược lựa chọn nhất quán với thiết kế , theo đó các mô hình tuyến tính và phi tuyến tính hoạt động tốt nhất được xác định riêng biệt trong mỗi phương án đánh giá.

Cách tiếp cận này được thúc đẩy bởi các phát hiện thực nghiệm trong phần trước, chứng minh rằng hiệu suất và thứ hạng của mô hình rất nhạy cảm với thiết kế đánh giá . Cụ thể, trong khi một số mô hình hoạt động tốt dưới phương pháp kiểm định chéo chuỗi thời gian (TSCV), thì những mô hình khác lại thể hiện hiệu suất vượt trội hơn trong khuôn khổ cửa sổ mở rộng trượt. Do đó, việc lựa chọn một cặp mô hình duy nhất trong tất cả các thiết lập có thể dẫn đến các đặc tả lai không tối ưu hoặc thiên lệch.

Theo đó, đối với mỗi khung đánh giá, chúng tôi chọn mô hình có hiệu suất cao nhất từ ​​lớp tuyến tính và lớp phi tuyến tính dựa trên hiệu suất ngoài mẫu. Do đó, các mô hình được chọn phản ánh sự thể hiện phù hợp nhất của các cấu trúc tuyến tính và phi tuyến tính trong từng thiết kế kiểm định cụ thể .

Sử dụng các cặp mô hình đặc thù cho từng thiết kế này, chúng tôi xây dựng hai mô hình lai bằng cách kết hợp các dự đoán của chúng thông qua các lược đồ trọng số dựa trên hiệu suất. Chiến lược này đảm bảo rằng mỗi đặc tả lai được căn chỉnh tối ưu với khung đánh giá cơ bản, từ đó cung cấp một đánh giá đáng tin cậy hơn về lợi ích của việc kết hợp mô hình.

#### Timeseries hybird

<img width="847" height="540" alt="image" src="https://github.com/user-attachments/assets/c414ea6c-8dbf-4f8d-9d2c-d1c2a49fc22b" />

### rolling 

<img width="855" height="547" alt="image" src="https://github.com/user-attachments/assets/6792c3ae-6092-4085-ac6f-7c78b25260a3" />

## feature important 
### cv 

<img width="1040" height="403" alt="image" src="https://github.com/user-attachments/assets/c892b64e-4591-4f79-b11d-8e51e14c9192" />

### rolling
<img width="1040" height="403" alt="image" src="https://github.com/user-attachments/assets/b0716987-785d-4925-b160-45f2030ca4d2" />

## SHAP 
### cv

<img width="1991" height="591" alt="image" src="https://github.com/user-attachments/assets/72fc2aaf-c88d-47ca-8fff-82bdd86264e2" />

<img width="762" height="755" alt="image" src="https://github.com/user-attachments/assets/8b3ed216-eb35-4f55-bd01-f40e7083e772" />

### rolling 

<img width="1991" height="591" alt="image" src="https://github.com/user-attachments/assets/016d5d72-faff-47b8-ada3-0ec5a2d9e1aa" />

<img width="762" height="741" alt="image" src="https://github.com/user-attachments/assets/19d649ec-df8e-4dff-8c6d-a7d26394234b" />

<img width="641" height="453" alt="image" src="https://github.com/user-attachments/assets/430f7545-c61b-4281-876c-ad889e69de1d" />
<img width="654" height="453" alt="image" src="https://github.com/user-attachments/assets/a86c3491-ea69-4928-95a8-e3c66b5c9249" />
<img width="666" height="453" alt="image" src="https://github.com/user-attachments/assets/65eae4c2-642a-4861-88f7-4f6ba874efac" />
<img width="674" height="453" alt="image" src="https://github.com/user-attachments/assets/02e74099-1fcd-4aca-925e-9d4a8d096854" />
<img width="666" height="453" alt="image" src="https://github.com/user-attachments/assets/ddac38a5-8c10-440e-a755-3b7577aea57f" />
<img width="666" height="453" alt="image" src="https://github.com/user-attachments/assets/2a75aa6a-cfab-4b94-8452-2afe3ee40263" />


| ![Image 1](https://github.com/user-attachments/assets/430f7545-c61b-4281-876c-ad889e69de1d) | ![Image 2](https://github.com/user-attachments/assets/77d28c8c-72b6-46ee-a784-da7eafe3830e) |
|:----------------------------------------------------------:|:-----------------------------------------------------------:|
| ![Image 7](https://github.com/user-attachments/assets/2a75aa6a-cfab-4b94-8452-2afe3ee40263)  | ![Image 4](https://github.com/user-attachments/assets/65eae4c2-642a-4861-88f7-4f6ba874efac) |
| ![Image 5](https://github.com/user-attachments/assets/02e74099-1fcd-4aca-925e-9d4a8d096854) | ![Image 6](https://github.com/user-attachments/assets/ddac38a5-8c10-440e-a755-3b7577aea57f) | 




# DML with LGBM
Lựa chọn lgbm để làm base cho DML là bởi vì độ ổn định mô hình thể hiện trong các kết quả metric ở cả phương pháp time sẻies cv hay là rolling. 
theo thứ tự các feature important  và shap important của lgbm , nghiên cứu này đã tiến hành phân tích treadment của các feature top đầu để xác nhận hiệu ứng nhân quả và ảnh hưởng tới output firm value của các feature mà shap đã giải thích hay không 


| Treatment   |   Theta |   Std.Error |   P-value |
|:------------|--------:|------------:|----------:|
| SIZE        |  0.2167 |      0.0083 |    0.0000 |
| Z_SCORE     |  0.9279 |      0.1201 |    0.0000 |
| LEV         |  0.7017 |      0.1119 |    0.0000 |
| GP          |  0.7128 |      0.0658 |    0.0000 |
| OPM         |  0.0845 |      0.0614 |    0.1691 |
| LIQ         |  0.0104 |      0.0035 |    0.0027 |
| CH          |  0.7959 |      0.1283 |    0.0000 |
| DIV_DUMMY   | -0.3586 |      0.0142 |    0.0000 |
| ROE          | -0.1303 |      0.1687 |    0.4400 |
| EBITDA_TA    |  0.3967 |      0.3357 |    0.2373 |
| NPM          |  0.0017 |      0.0035 |    0.6319 |
| TANG         | -0.0340 |      0.0460 |    0.4608 |
| ROA          |  0.4228 |      0.2900 |    0.1449 |
| GROWTH_SALES |  0.0013 |      0.0027 |    0.6196 |
| FCF_TA       |  0.0442 |      0.0491 |    0.3674 |

<img width="1187" height="690" alt="image" src="https://github.com/user-attachments/assets/5311ffef-6e77-4eb2-99ad-081500aa041f" />

| Term       |    coef |   std err |       t |   P>|t| |   2.5 % |   97.5 % | Treatment_Base   |
|:-----------|--------:|----------:|--------:|--------:|--------:|---------:|:-----------------|
| OPM        |  0.0619 |    0.0569 |  1.0876 |  0.2768 | -0.0555 |   0.1735 | OPM              |
| OPM_sq     | -0.0039 |    0.0024 | -1.6444 |  0.1001 | -0.0086 |   0.0007 | OPM              |
| LEV        |  0.4553 |    0.0958 |  4.7512 |  0.0000 |  0.2675 |   0.6432 | LEV              |
| LEV_sq     |  0.1632 |    0.0990 |  1.6476 |  0.0994 | -0.0303 |   0.3573 | LEV              |
| Z_SCORE    |  0.6909 |    0.0870 |  7.9372 |  0.0000 |  0.5252 |   0.8615 | Z_SCORE          |
| Z_SCORE_sq |  0.0601 |    0.0121 |  4.9746 |  0.0000 |  0.0355 |   0.0837 | Z_SCORE          |

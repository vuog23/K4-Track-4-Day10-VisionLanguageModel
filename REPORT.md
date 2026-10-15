Họ tên: Nguyễn Triều Vương
`dataset_version`: `d10-21e4070301` (theo `report/evidence/dataset_stats.json` và `work/val_subset/splits/meta.json`)

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup                                  | Tham số huấn luyện | VRAM đỉnh (GB) | Phút             | F1    | CI95           | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
| --- | -------------------------------------- | --------------------- | ---------------- | ----------------- | ----- | -------------- | ------- | ---------- | -------- | ---------- |
| E0  | zero-shot                              | 0                     | 3,77             | 21,06 (suy luận) | 0,074 | [0,051; 0,099] | 1,31    | 0,089      | 0,020    | 0,179      |
| E1  | LoRA LLM                               | 33.030.144            | 6,26             | 41,01             | 0,269 | [0,230; 0,308] | 1,38    | 0,189      | 0,187    | 0,561      |
| E2  | + projector                            | 142.135.296           | 7,87             | 45,71             | 0,356 | [0,316; 0,400] | 1,14    | 0,247      | 0,277    | 0,628      |
| E3  | + LoRA ViT                             | 148.426.752           | 8,04             | 50,97             | 0,416 | [0,374; 0,455] | 1,00    | 0,315      | 0,342    | 0,653      |
| E4  | E3, `data_size=0.1` (142 ảnh train) | 148.426.752           | 8,03             | 51,15             | 0,251 | [0,219; 0,284] | 1,89    | 0,179      | 0,171    | 0,531      |

F1 là micro-F1@IoU 0,5 trên 311 ảnh test. Cột phút của E1–E4 là **thời gian huấn luyện**, không gồm suy luận/chấm điểm; E0 không huấn luyện nên ghi thời gian suy luận. E4 mức 100% dùng lại kết quả E3, không huấn luyện thêm một lượt.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Theo `report/evidence/model_parameters.json`: ViT **306.242.560**, projector **109.105.152**, LLM cùng embedding/head **4.022.468.096**, tổng **4.437.815.808** tham số trước khi xét dung lượng lượng tử hoá. E1 chỉ cập nhật LoRA trên attention/MLP của LLM (33.030.144 tham số). E2 thêm toàn bộ projector, nâng số tham số cập nhật lên 142.135.296. E3 thêm LoRA trên attention/MLP của ViT, tổng 148.426.752. Trọng số gốc của ViT và LLM vẫn đóng băng.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

`parse_valid_rate` của E0 là **0,7717** (240/311 ảnh hợp lệ; 71 ảnh không parse được). Hai lỗi thấy trực tiếp trong `work/e0.jsonl` là: (1) model lặp nhiều box cho vết nứt cho đến khi JSON bị cắt, chẳng hạn `vlcsnap-00209` bắt đầu bằng `[{"label": "crack", "bbox_2d": [500, 670, 520, 690]}, ...` nhưng không kết thúc thành JSON hợp lệ; (2) lặp nguyên box trong một output hợp lệ, chẳng hạn `vlcsnap-00202`: `[{"label": "manhole", "bbox_2d": [309, 686, 399, 714]}, {"label": "manhole", "bbox_2d": [309, 686, 399, 714]}]`. Lỗi thứ hai tạo dự đoán thừa; E0 có 407 FP và F1 chỉ 0,074. Đây là hai dạng lỗi quan sát được, không phải bảng xếp hạng nguyên nhân lỗi độc lập.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Giữ nguyên dự đoán E0 nhưng chấm nhầm với `coord=pixel` làm micro-F1 từ **0,0739** xuống **0**, TP từ 38 xuống 0 (`e0.json` so với `e0_wrong_coord_pixel.json`). Bộ chấm sẽ chia các toạ độ dự đoán cho kích thước ảnh 640×360 rồi nhân 1000, trong khi output đã ở thang 0–1000; box bị co/lệch nên không đạt IoU 0,5 với ground truth. Đây là kiểm chứng bằng **chấm lại cùng output**, không phải chạy lại model.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Mục tiêu huấn luyện là sinh JSON sau lời nhắc và ảnh. Trong `encode_example`, notebook đặt `labels=-100` cho toàn bộ prefix gồm prompt và token ảnh, đồng thời mask phần padding; chỉ token câu trả lời đi vào cross-entropy. Nếu tính loss trên cả prefix, model phải học dự đoán lại lời nhắc cố định và chuỗi token ảnh, làm mục tiêu tối ưu lệch khỏi định dạng/toạ độ JSON cần sinh. Bộ dữ liệu này chưa có thí nghiệm đối chứng huấn luyện không mask, nên đây là giải thích theo mục tiêu và code, không phải hiệu ứng F1 đã đo.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

Nếu lượng tử hoá cả `model.visual`, các trọng số projector bên trong nhánh này có thể thành tensor 4-bit, không thể bật `requires_grad` để cập nhật đầy đủ như E2/E3. Notebook đặt `llm_int8_skip_modules=["model.visual", "lm_head"]`, kiểm tra projector vẫn có khoảng 109 triệu tham số, rồi chuyển tham số projector cần học sang **float32** trước khi bật gradient. Float32 ở đây giúp gradient/optimizer của projector tương thích với vòng huấn luyện dùng `GradScaler` và giảm nguy cơ tràn số; các khối LLM vẫn dùng NF4, ViT gốc ở fp16. Về nguyên lý có thể lượng tử hoá riêng ViT mà giữ projector không lượng tử, nhưng đó không phải cấu hình đã chạy.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Các ảnh là frame cùng phiên quay nên frame gần nhau thường gần trùng cảnh. Chia ngẫu nhiên từng ảnh có thể đưa các frame gần giống vào cả train và test, làm F1 test cao giả vì rò rỉ ngữ cảnh. Split cố định theo nhóm phiên quay trong `meta.json` cho:

| Split |  Ảnh | Box pothole | Box crack | Box manhole |
| ----- | ----: | ----------: | --------: | ----------: |
| Train | 1.422 |       1.074 |     1.750 |         691 |
| Val   |   276 |          66 |       470 |         103 |
| Test  |   311 |         121 |       299 |         163 |

Chưa có đối chứng chia ngẫu nhiên, nên mức F1 tăng giả không được định lượng.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

E1 đạt F1 **0,269**, CI95 **[0,230; 0,308]**, VRAM đỉnh **6,26 GB**, train **41,01 phút**. Mở projector ở E2 nâng F1 lên **0,356** (+0,088 điểm), VRAM lên **7,87 GB** (+1,60 GB), thời gian lên **45,71 phút** (+4,71 phút). Thêm LoRA ViT ở E3 nâng F1 lên **0,416** (+0,060 so với E2), VRAM **8,04 GB** (+0,18 GB), thời gian **50,97 phút** (+5,26 phút); CI95 E2 **[0,316; 0,400]**, E3 **[0,374; 0,455]**. Trên lượt chạy này, cả hai bước mở thêm khối đều cải thiện F1, còn E3 có điểm cao nhất. Các con số thời gian là thời gian train từng run; chạy đồng thời hai GPU không đồng nghĩa cộng các thời gian này để ra wall time toàn notebook.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

CI95 bootstrap micro-F1 trên 311 ảnh test rộng khoảng **0,077** (E1), **0,084** (E2) và **0,081** (E3); E0 rộng khoảng **0,048**. E2 và E3 có khoảng giao nhau **[0,374; 0,400]**. Vì vậy không nên tuyên bố chắc chắn E3 tốt hơn E2 ngoài mẫu test chỉ từ các CI riêng lẻ; cần CI bootstrap của **chênh lệch ghép cặp** hoặc thêm phiên quay độc lập. Việc chọn E3 trước E4 đã dựa trên tập val 40 ảnh (`selection_e4.json`: F1 E1 0,191; E2 0,304; E3 0,349), còn test chỉ dùng để báo cáo kết quả.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

Với setup E3 được chọn từ val, **pothole** có F1 thấp nhất: **0,315** (34 TP, 61 FP, 87 FN), so với crack **0,342** và manhole **0,653**. Box pothole ở train có diện tích trung vị chỉ **0,376%** ảnh; test có **121 box** pothole, ít hơn crack (**299**) và manhole (**163**). Kích thước nhỏ và số mẫu test ít có thể làm việc nhận diện/định vị khó hơn; bề mặt ổ gà cũng dễ lẫn với mảng đường tối hoặc hư hại không có ranh giới rõ. Phần ngữ nghĩa này là giả thuyết từ ảnh minh hoạ, chưa được kiểm chứng bằng phân loại lỗi thủ công.

![Ví dụ pothole: GT màu xanh, dự đoán E3 màu đỏ](report/figures/q9_e3_examples.png)

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

E4 đổi **kích thước dữ liệu train** trên cấu hình E3: 10% (**142 ảnh**) so với 100% (**1.422 ảnh**, dùng lại E3). Cả hai vẫn chạy 300 bước. F1 test giảm từ **0,416** xuống **0,251** (−0,165 điểm), FP/ảnh tăng từ **1,00** lên **1,89**; F1 val-40 giảm từ **0,349** xuống **0,158**. CI95 test lần lượt là **[0,374; 0,455]** và **[0,219; 0,284]**. Kết quả ủng hộ dùng nhiều ảnh hơn trong thiết lập này, nhưng chỉ có một lần lấy mẫu 10%, một seed và một tập test theo phiên quay; không suy ra chính xác độ lớn lợi ích cho mọi dữ liệu. Vì cùng 300 bước, nhóm 10% lặp lại ảnh nhiều hơn; thời gian train gần như không giảm (51,15 so với 50,97 phút).

## Nhìn lại

E0 tạo JSON bị cắt do lặp box và cả JSON hợp lệ chứa box trùng; `parse_valid_rate` chỉ bắt loại đầu, còn loại sau đi vào chấm điểm như dự đoán thừa. E3 còn **3 bước cập nhật bị bỏ qua do tràn gradient** (`overflow_skips` trong `work/e3_fp16vision/run.json`) dù cuối cùng đủ 300 bước. Muốn phân biệt cải thiện thật với dao động do seed/phiên quay, cần thêm lần chạy độc lập và CI ghép cặp cho chênh lệch F1; hiện chưa có các bằng chứng đó.

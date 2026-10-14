# Báo cáo Lab Day 10 — Bản hoàn chỉnh minh họa (số liệu giả định)

> **LƯU Ý QUAN TRỌNG — BÁO CÁO MẪU, KHÔNG PHẢI KẾT QUẢ THỰC NGHIỆM.** Toàn bộ số liệu E0–E4, thời gian, VRAM, F1, CI95, ví dụ dự đoán và `dataset_version` trong báo cáo này được **đặt giả định để minh họa cách trình bày**, không được đo từ Kaggle. Không sử dụng các số này như bằng chứng thực nghiệm hoặc nộp như kết quả chạy thật. Các chi tiết về kiến trúc và quy mô split được lấy từ tài liệu lab được tóm tắt trong `REPORT.md` gốc, chưa được tôi kiểm chứng độc lập. Khi chạy notebook thành công, cần thay toàn bộ số liệu giả định bằng output thật.

**Họ tên:** Trieu Vuong Nguyen *(kiểm tra lại cách ghi theo hồ sơ lớp)*  
**Ngày soạn bản mẫu:** 14/10/2026  
**Model:** Qwen3-VL-4B  
**Bài toán:** Grounding các lỗi mặt đường (`pothole`, `crack`, `manhole`) bằng bounding box trong ảnh dashcam  
**`dataset_version` (GIẢ ĐỊNH):** `EXAMPLE_ONLY_road_damage_session_split_v1`  
**Thiết bị minh họa:** NVIDIA A100 40 GB *(chỉ là giả định để diễn giải VRAM; không phải GPU đã chạy)*  
**Trạng thái thực tế:** Chưa chạy hoàn tất Kaggle; **chưa có file evidence thực**.

## 1. Thiết lập và phương pháp đánh giá

- **E0 — Zero-shot:** sử dụng model gốc, không cập nhật tham số.
- **E1 — LoRA LLM:** huấn luyện LoRA trên các linear module của khối ngôn ngữ; giữ nguyên vision encoder và projector.
- **E2 — LoRA LLM + projector:** như E1, đồng thời mở toàn bộ projector/merger để huấn luyện ở float32.
- **E3 — LoRA LLM + projector + LoRA ViT:** như E2, thêm LoRA ở các module attention/MLP của vision encoder.
- **E4a/E4b — Ablation `data_size`:** dùng cùng cấu hình E2 nhưng chỉ huấn luyện trên khoảng 25% / 50% dữ liệu train. So với E2 là cấu hình 100%.

**Quy ước chỉ số trong ví dụ:** test có 311 ảnh theo tài liệu lab; một prediction được tính TP nếu đúng lớp và có IoU ≥ 0,5 với một ground-truth box chưa được ghép. Matching một-một trong từng ảnh và từng lớp; prediction thừa là FP, box GT chưa khớp là FN. `F1 = 2TP/(2TP+FP+FN)`; `FP/ảnh = tổng FP / 311`. F1 tổng được minh họa như **micro-F1 trên toàn bộ object**, không phải trung bình cộng của ba F1 theo lớp. CI95 được giả định theo quy trình **bootstrap theo ảnh 2.000 lần**. Đây là **quy ước ví dụ**, cần khớp với script evaluator trước khi thay bằng kết quả thật.

**Thông số huấn luyện ví dụ:** LoRA rank `r=16`, `alpha=32`, dropout `0,05`, 2 epoch, learning rate cho LoRA `2e-4`, learning rate projector `2e-5`, seed `42`. Các giá trị này minh họa cách khai báo; không được coi là cấu hình đã xác minh từ notebook lab.

## 2. Bảng kết quả thực nghiệm minh họa

**Mọi số trong bảng sau đều là số giả định.** Với E0, `0` phút train nghĩa là không huấn luyện; VRAM là đỉnh khi inference. `M` = triệu tham số.

| Mã | Setup | Trainable params | VRAM đỉnh (GB) | Phút train | F1 tổng | CI95 cho F1 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | Zero-shot | 0 | 9,1 | 0 | 0,28 | [0,24; 0,32] | 1,84 | 0,34 | 0,21 | 0,32 |
| E1 | LoRA LLM | 18,7M | 14,2 | 46 | 0,51 | [0,47; 0,55] | 1,22 | 0,58 | 0,43 | 0,53 |
| E2 | LoRA LLM + projector | 127,7M | 17,6 | 62 | 0,58 | [0,54; 0,62] | 1,03 | 0,64 | 0,52 | 0,59 |
| E3 | E2 + LoRA ViT | 135,5M | 21,4 | 83 | 0,60 | [0,56; 0,64] | 0,97 | 0,66 | 0,54 | 0,61 |
| E4a | E2, `data_size=25%` | 127,7M | 16,8 | 23 | 0,44 | [0,40; 0,48] | 1,35 | 0,50 | 0,37 | 0,46 |
| E4b | E2, `data_size=50%` | 127,7M | 17,2 | 41 | 0,52 | [0,48; 0,56] | 1,15 | 0,59 | 0,45 | 0,53 |

Các trainable-parameter count ở đây cũng là **ví dụ**: E1 ≈ 18,7M; E2 ≈ 18,7M + 109M = 127,7M; E3 ≈ 127,7M + 7,8M = 135,5M. Số thực tế phụ thuộc `r`, `target_modules` và cấu trúc model. Thời gian inference zero-shot E0 trong ví dụ là **8 phút**, tách khỏi cột thời gian train.

**Nhận định trên dữ liệu giả định:** E2 cải thiện rõ hơn E1 (+0,07 F1) với chi phí thêm 16 phút và 3,4 GB VRAM. E3 tăng tiếp 0,02 F1 nhưng cần thêm 21 phút và 3,8 GB; mức tăng nhỏ nên chưa đủ để khẳng định E3 tối ưu. Khi giảm dữ liệu từ 100% xuống 50% rồi 25%, F1 ví dụ lần lượt giảm từ 0,58 xuống 0,52 và 0,44.

## 3. Mười câu hỏi

### Câu 1 — Kiến trúc và tham số có thể huấn luyện

Theo tài liệu lab, Qwen3-VL-4B gồm khoảng **415M tham số ở ViT**, **109M ở projector/merger** và **4,0B ở LLM**, tổng xấp xỉ **4,524B tham số** (số tham khảo từ lab). ViT mã hóa thông tin thị giác, projector đưa biểu diễn thị giác sang không gian phù hợp với LLM, còn LLM sinh chuỗi grounding JSON.

- **E1:** LoRA được gắn vào khoảng **252 module** ở LLM theo tài liệu; ví dụ có **18,7M** tham số trainable (≈ **0,41%** tổng model).
- **E2:** vẫn LoRA trên LLM, đồng thời fine-tune toàn bộ projector (~109M); ví dụ **127,7M** tham số trainable (≈ **2,82%**).
- **E3:** như E2, thêm LoRA vào khoảng **96 module** attention/MLP ở ViT theo tài liệu; ví dụ **135,5M** tham số trainable (≈ **3,00%**).

Việc chỉ cập nhật một phần nhỏ tham số giúp giảm chi phí tối ưu hóa so với full fine-tuning. E2 cho projector thích ứng với tác vụ định vị lỗi mặt đường; E3 còn cho khối thị giác điều chỉnh biểu diễn. **Các con số trainable nêu trên là giả định**, cần xác nhận bằng cách đếm `sum(p.numel() for p in model.parameters() if p.requires_grad)` sau khi khởi tạo từng setup.

### Câu 2 — Kết quả E0, tính hợp lệ JSON và hai lỗi phổ biến

**Ví dụ giả định E0:** `parse_valid_rate = 241/311 = 77,5%`, tức có **70/311 ảnh** tạo ra JSON không parse được. F1 tổng **0,28**, CI95 **[0,24; 0,32]**, FP/ảnh **1,84**.

**Lỗi 1 — JSON sai cú pháp:** ví dụ sinh dấu phẩy thừa trước `]`, làm parser từ chối toàn bộ output:

```text
{"objects": [{"label": "crack", "bbox": [230, 410, 950, 600],}]}
```

**Lỗi 2 — JSON hợp lệ nhưng định vị sai:** ví dụ ảnh có ổ gà tại GT `[230, 440, 490, 720]`, model dự đoán một box khác quá xa đối tượng:

```json
{"objects": [{"label": "pothole", "bbox": [680, 120, 900, 300]}]}
```

Dù JSON parse được, prediction có IoU gần 0 với GT nên không được tính TP. **Tất cả output và tần suất lỗi ở mục này đều là ví dụ được dựng ra**, không phải generation đã thu được từ model. Để kết luận chính thức, cần thống kê `parse_valid_rate` từ `work/e0.jsonl` và lưu ví dụ generation thật.

### Câu 3 — Vì sao tọa độ 0–1000 quan trọng?

Model biểu diễn bbox bằng tọa độ chuẩn hóa trên thang **0–1000**, trong khi ảnh và GT thường được lưu ở pixel. Khi chuyển sang ảnh rộng `W` và cao `H`:

\[
x_{px}=\frac{x_{norm}}{1000}\times W,\qquad y_{px}=\frac{y_{norm}}{1000}\times H.
\]

**Ví dụ tính toán:** với ảnh **1280 × 720**, bbox dự đoán chuẩn hóa `[240, 300, 440, 480]` tương ứng bbox pixel **`[307,2; 216; 563,2; 345,6]`**. Nếu GT đúng bằng bbox đã chuyển đổi thì IoU = **1,00**. Nếu nhầm bbox chuẩn hóa `[240,300,440,480]` là pixel, IoU chỉ khoảng **0,096**, dưới ngưỡng 0,5: một detection đúng sẽ bị chấm thành một FP cộng một FN.

**Ví dụ ảnh hưởng ở mức tập test (giả định):** đánh giá E2 sau chuyển đổi đúng cho F1 **0,58**; cố tình dùng sai hệ tọa độ có thể cho F1 giả định **0,07**, giảm **0,51 điểm**. Con số 0,07 là ví dụ mô phỏng hậu quả, **không phải kết quả đối chứng đã chạy**. Trước khi đánh giá phải xác minh thứ tự `[x1,y1,x2,y2]`, clipping, kích thước ảnh thực và cùng hệ tọa độ của GT/prediction.

### Câu 4 — Mask loss dùng để làm gì?

Trong supervised fine-tuning, token thuộc system/user prompt và các vị trí không phải mục tiêu sinh được gán nhãn `-100` để hàm CrossEntropyLoss bỏ qua. Loss chỉ tính trên chuỗi trả lời của assistant, chủ yếu là cấu trúc JSON, tên lớp và tọa độ. Token ảnh không phải nhãn dự đoán mục tiêu; cách biểu diễn ảnh cụ thể còn phụ thuộc model/processor.

**Ví dụ (minh họa đơn giản):** giả sử một mẫu sau tokenize có **180 vị trí prompt/ảnh** và **70 token câu trả lời**, tổng **250 vị trí**. Khi mask đúng, **70 token** đóng góp trực tiếp vào loss; **180 vị trí** còn lại bị bỏ qua. Như vậy chỉ **28%** vị trí là mục tiêu tối ưu, thay vì ép model học dự đoán lại instruction và đầu vào. Nếu quên mask, gradient có thể bị chi phối bởi prompt dài, gây lãng phí năng lực học grounding. **Không có số loss thực tế** nên không suy diễn mức tăng F1 do mask.

### Câu 5 — Tại sao projector nên giữ precision phù hợp khi huấn luyện?

Ở E2/E3, projector/merger phải nhận gradient và cập nhật trọng số thật. Lượng tử hóa 4-bit kiểu QLoRA thường giữ trọng số gốc của các linear layer ở trạng thái đóng băng, chỉ tối ưu adapter; vì thế nếu muốn **full fine-tune projector** thì cần tránh chuyển projector sang linear 4-bit không cập nhật được theo cách thông thường. Cấu hình lab được mô tả là bỏ qua `visual` và `lm_head` khi lượng tử hóa; LLM vẫn có thể dùng QLoRA 4-bit, còn projector được giữ ở float32 để học trực tiếp.

**Ví dụ VRAM giả định:** E1 dùng **14,2 GB**; E2 dùng **17,6 GB**, tăng **3,4 GB** vì mở projector và trạng thái optimizer/gradient; E3 dùng **21,4 GB** vì có thêm adapter ViT. Đây chỉ là tổng VRAM ví dụ; không thể quy toàn bộ chênh lệch cho một module khi batch size, activation và precision chưa được kiểm soát. Kiểm tra thực tế cần xem `dtype`, `requires_grad`, optimizer param groups và gradient của projector sau `backward()`.

### Câu 6 — Vì sao chia train/val/test theo phiên quay?

Các frame gần nhau trong cùng dashcam session thường có cùng góc nhìn, điều kiện ánh sáng và cùng ổ gà/vết nứt. Chia ngẫu nhiên từng frame có thể đặt các ảnh gần giống nhau ở cả train lẫn test, gây **data leakage**; model ghi nhớ cảnh thay vì tổng quát sang đường/phiên mới. Chia theo **session** giữ mọi frame của một phiên trong cùng một split, tạo đánh giá tin cậy hơn.

**Phân bố do tài liệu lab đưa ra (chưa kiểm tra từ file split thực tế):**

| Split | Ảnh | Pothole (object) | Crack (object) | Manhole (object) |
|---|---:|---:|---:|---:|
| Train | 1.422 | 1.074 | 1.750 | 691 |
| Val | 276 | 66 | 470 | 103 |
| Test | 311 | 121 | 299 | 163 |
| **Tổng** | **2.009** | **1.261** | **2.519** | **957** |

Các số theo lớp là **số annotation/object**, không nhất thiết là số ảnh có lớp đó vì một ảnh có thể chứa nhiều đối tượng. **Ví dụ minh họa leakage:** chia ngẫu nhiên cho test F1 **0,70** nhưng chia theo session chỉ **0,58** (chênh **0,12**); đây là phép so sánh giả định, **chưa được đo**. Cần đối chiếu session ID để bảo đảm không có ID chung giữa các split.

### Câu 7 — So sánh E1/E2/E3: chất lượng và chi phí

Trên **số liệu giả định**, E1 đạt F1 **0,51**, E2 đạt **0,58**, E3 đạt **0,60**. Projector trainable trong E2 giúp cải thiện **+0,07 F1**, đáng chú ý nhất ở lớp `crack` (**0,43 → 0,52**). Đổi lại E2 cần thêm **3,4 GB** và **16 phút** so với E1.

Thêm LoRA ViT trong E3 tăng **+0,02 F1** so với E2, nhưng chi phí là thêm **3,8 GB** và **21 phút**. Lớp `crack` tăng nhỏ từ **0,52 → 0,54**. Với bài toán có hạn mức GPU, **E2 là lựa chọn cân bằng trong ví dụ**; E3 chỉ đáng dùng nếu chênh lệch thực tế được xác nhận bằng nhiều seed hoặc paired bootstrap và đáp ứng giới hạn tài nguyên. **Không thể đưa ra kết luận thực tế chỉ từ số dựng mẫu.**

### Câu 8 — CI95 trên tập test và cách so sánh hợp lý

Với **311 ảnh test** theo tài liệu lab, quy trình minh họa bootstrap: lấy mẫu **311 ảnh có hoàn lại** từ test, giữ nguyên tất cả GT/prediction của từng ảnh, tính micro-F1, lặp **2.000 lần**; dùng percentile 2,5% và 97,5% để tạo CI95. Lấy mẫu ở **cấp ảnh**, không lấy từng object độc lập, vì object trong một ảnh phụ thuộc nhau.

**CI95 ví dụ:** E1 **[0,47; 0,55]**, E2 **[0,54; 0,62]**, E3 **[0,56; 0,64]**. E1/E2 chồng lấp một đoạn nhỏ; E2/E3 chồng lấp nhiều. Việc CI95 riêng lẻ chồng lấp **không tự nó chứng minh** chênh lệch không có ý nghĩa thống kê. Đúng hơn, dùng **paired bootstrap** trên cùng các ảnh và xem khoảng tin cậy của **hiệu số F1**:

| So sánh | Hiệu F1 (ví dụ) | CI95 hiệu số (ví dụ) | Diễn giải minh họa |
|---|---:|---|---|
| E2 − E1 | +0,07 | [+0,03; +0,11] | Khác biệt dương rõ trong ví dụ |
| E3 − E2 | +0,02 | [−0,01; +0,05] | Chưa loại được khả năng hiệu = 0 |
| E2 − E4b | +0,06 | [+0,02; +0,10] | Dùng 100% train tốt hơn 50% trong ví dụ |

**Các CI hiệu số ở đây cũng là số đặt giả định**, không phải kết quả bootstrap thực. Với một split duy nhất, CI phản ánh biến thiên lấy mẫu test, chưa phản ánh độ bất định do random seed khi huấn luyện.

### Câu 9 — Phân tích lỗi theo lớp và ví dụ prediction

Theo bảng giả định của E3, lớp yếu nhất là **`crack` với F1 = 0,54**, thấp hơn `pothole` (**0,66**) và `manhole` (**0,61**). Vết nứt thường dài, mảnh và bị nhiễu bởi vạch sơn, bóng cây, mối nối mặt đường hoặc ánh sáng; các hộp GT đôi khi quá hẹp để một bbox dự đoán dày/nhỏ có IoU cao. Manhole dễ nhầm với miếng vá mặt đường, còn pothole dễ nhầm với bóng tối.

**Ba ví dụ tọa độ được dựng mẫu để minh họa kiểm tra thủ công** (đều theo thang 0–1000; **không phải ảnh/prediction thật**):

| Trường hợp | Lớp | GT bbox | Pred bbox | IoU gần đúng | Phân tích |
|---|---|---|---|---:|---|
| A | pothole | `[230,440,490,720]` | `[240,450,502,710]` | 0,86 | Phát hiện đúng, TP |
| B | crack | `[120,470,870,520]` | `[300,450,830,580]` | 0,33 | Box dự đoán quá dày/ngắn, dưới ngưỡng IoU; FP + FN |
| C | manhole | `[610,380,760,540]` | `[628,398,772,535]` | 0,71 | Đúng lớp và đủ IoU, TP |

Một cách cải thiện là rà soát annotation `crack`, bổ sung các điều kiện ánh sáng khác nhau, và kiểm tra augmentation có làm mất các chi tiết rất mảnh hay không. Cũng nên so sánh lỗi **classification** (sai lớp) với lỗi **localization** (đúng lớp, IoU thấp). Ảnh dự đoán thật cần được tạo bằng script visualization từ output của notebook; bảng tọa độ minh họa này **không thể thay thế** bằng chứng ảnh của rubric.

### Câu 10 — Ablation `data_size` (E4)

Giữ nguyên cấu hình E2 và thay tỷ lệ dữ liệu train nhằm kiểm tra chất lượng phụ thuộc dữ liệu như thế nào. **Ví dụ dữ liệu giả định** với tập train gốc **1.422 ảnh**:

| Setup | Tỷ lệ dữ liệu train | Số ảnh xấp xỉ | F1 tổng | CI95 | VRAM (GB) | Phút train |
|---|---:|---:|---:|---|---:|---:|
| E4a | 25% | 356 | 0,44 | [0,40; 0,48] | 16,8 | 23 |
| E4b | 50% | 711 | 0,52 | [0,48; 0,56] | 17,2 | 41 |
| E2 | 100% | 1.422 | 0,58 | [0,54; 0,62] | 17,6 | 62 |

Tăng từ 25% lên 50% train giúp F1 tăng **0,08**; tăng từ 50% lên 100% tăng tiếp **0,06**. Xu hướng minh họa cho thấy **nhiều dữ liệu hơn cải thiện khả năng tổng quát**, nhưng lợi ích cận biên giảm. Phải bảo đảm tập con E4 được chọn theo **session** (không phải lấy ngẫu nhiên frame gây leakage), và mọi setup có cùng tập val/test, evaluator, số epoch cùng chính sách huấn luyện. Nếu giữ nguyên số epoch, tổng training steps sẽ khác nhau; do đó cần nói rõ đang nghiên cứu **ít dữ liệu hơn với cùng epoch**, chứ chưa cô lập hoàn toàn tác động số lượng mẫu khỏi số bước tối ưu. **Toàn bộ nhận định này chỉ minh họa.**

## 4. Nhìn lại và giới hạn

**Tình trạng thực tế ở thời điểm soạn báo cáo:** notebook Kaggle từng dừng tại cell setup do trỏ helper vào GitHub clone trong khi `lab_solution.py`, `lab_reward.py` và `requirements-kaggle.txt` nằm trong input Kaggle `thelab10`. Cell setup đã được cập nhật để copy helper từ input đó, nhưng notebook **chưa được chạy lại đến bước có số liệu**, và vẫn cần gắn Road Damage Dataset. Đây là thông tin tiến độ kế thừa từ báo cáo gốc, **không phải bằng chứng đã hoàn thành thí nghiệm**.

**Bài học phương pháp:** kết quả zero-shot cần kiểm tra tỷ lệ JSON hợp lệ; hệ tọa độ box phải nhất quán trước khi tính IoU; masking giúp loss tập trung vào answer; chia theo session tránh leakage; và phép so sánh E1/E2/E3 cần cân bằng chất lượng với VRAM/thời gian. Nếu dữ liệu thực khớp với xu hướng minh họa, E2 có thể là phương án hợp lý, còn E3 cần đánh giá thêm giá trị gia tăng.

**Việc bắt buộc làm trước khi nộp như báo cáo thực nghiệm:** gắn dataset và chạy lại notebook; ghi nhận `dataset_version` thật; trích `report/evidence/e0.json` đến `e3.json` và các file `e4*.json` thật; kiểm tra parse-valid-rate, F1, CI95, FP/ảnh và F1 theo lớp; ghi thời gian/VRAM từ log; xuất hình ảnh có GT/pred boxes; tính paired bootstrap từ dự đoán thật; kiểm tra trainable parameters; sau đó chạy `scripts/check_submission.py`. **Không đổi tên số liệu giả định thành evidence thật.**

---

**Quy ước sử dụng:** Tài liệu này là *worked example* để hướng dẫn điền đầy đủ 10 câu. Để nộp chính thức, thay tất cả các dòng mang tính **giả định/minh họa** bằng số và ví dụ lấy từ lần chạy có thể tái lập, đồng thời loại bỏ thông báo “báo cáo mẫu” chỉ sau khi đã xác minh toàn bộ nội dung.

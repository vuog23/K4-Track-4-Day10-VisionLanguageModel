# Hướng dẫn thực hiện Lab Day 10

> Đọc [`README.md`](README.md) trước. Mỗi bước gồm **Mục tiêu · Pseudo code · Kiểm tra · Câu hỏi** (điền vào [`REPORT.md`](REPORT.md)). Pseudo code **không chạy được ngay**: bạn tự viết mã thật trong notebook (hoặc file `.py` của bạn).
> Quy ước: tên module lấy từ cây model Qwen3-VL-4B thật. Hãy tự in `[n for n, _ in model.named_modules()]` để xác nhận.

| Phút | Nội dung |
|---|---|
| 0–30 | Bước 0–1: môi trường, dữ liệu |
| 30–60 | Bước 2–3: nhãn JSON, E0 zero-shot |
| 60–90 | Bước 4–5: dữ liệu huấn luyện, setup LoRA |
| 90–200 | Bước 6–7: huấn luyện và chấm E1, E2, E3 |
| 200–230 | Bước 8: E4 |
| 230+ | Bước 9 (tuỳ chọn) và báo cáo |

---

## Bước 0 · Môi trường

Trong Kaggle Notebook, bật Internet và chọn **Settings → Accelerator → GPU T4 x2**. Cài thư viện theo lệnh sau; PyTorch/CUDA được cung cấp sẵn trong môi trường GPU.

```bash
pip install -r requirements-kaggle.txt
python -m pytest -q
```

Nếu gặp `ImportError ... requires the Torchvision library`, cài `torchvision` tương thích với bản torch của Kaggle image.

## Bước 1 · Dữ liệu

```bash
python scripts/prepare_data.py --source /kaggle/input/<dataset-folder>
```

Trước đó, dùng **Add Input** để gắn dataset Road Damage của lab. Notebook tự tìm thư mục chứa `images/` và `labels-YOLO/` trong `/kaggle/input`.

Sinh `data/images/`, `data/splits/{train,val,test}.jsonl` và `data/splits/meta.json`. Mỗi dòng `.jsonl`:

```json
{"image_id": "...", "image_path": "images/....jpg", "width": 640, "height": 360,
 "session": "vlc-2025-02-18-23", "objects": [{"label": "pothole", "yolo": [xc, yc, w, h]}]}
```

**Kiểm tra:** in số ảnh/box từng split; vẽ 6 ảnh có box lên để chắc bạn hiểu nhãn. Xác nhận không `session` nào xuất hiện ở hai split.

**Câu hỏi:** vì sao chia theo `session` mà không chia ngẫu nhiên? Điều gì xảy ra với F1 nếu chia ngẫu nhiên?

## Bước 2 · Nhãn YOLO → JSON thang 0–1000

```
hàm yolo_to_box(label, yolo):
    xc, yc, w, h = yolo                          # chuẩn hoá [0,1], tâm + kích thước
    x1 = (xc - w/2) * 1000;   y1 = (yc - h/2) * 1000
    x2 = (xc + w/2) * 1000;   y2 = (yc + h/2) * 1000
    làm tròn thành số nguyên, kẹp vào [0, 1000]
    trả {"label": label, "bbox_2d": [x1, y1, x2, y2]}

hàm objects_to_answer(objects):
    boxes = [yolo_to_box(...) cho mỗi object]
    sắp boxes theo (y1, x1)                      # trên→dưới, trái→phải: thứ tự ổn định cho model học
    trả json.dumps(boxes, separators=(",", ":")) # "[]" nếu không có object
```

Viết **prompt** yêu cầu model chỉ trả JSON list, nêu 3 lớp và nói rõ "thang 0–1000".

**Kiểm tra:** với một ảnh, chuyển nhãn sang JSON rồi quy ngược về pixel và vẽ lại; box phải trùng box gốc.

**Câu hỏi:** vì sao Qwen3-VL dùng thang 0–1000 chứ không dùng pixel tuyệt đối? (Slide 04, "đầu ra grounding").

## Bước 3 · E0: zero-shot

```
quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=float16, bnb_4bit_use_double_quant=True,
                  llm_int8_skip_modules=["visual", "lm_head"])
model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", quantization_config=quantization,
            dtype=float16, device_map={"": 0})
processor = AutoProcessor.from_pretrained(...);  processor.tokenizer.padding_side = "left"

với mỗi batch ảnh test (batch 4):
    messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text",  "text": PROMPT}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", padding=True)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = decode(out[:, độ_dài_prompt:], skip_special_tokens=True)
    ghi {"image_id": ..., "raw": raw} vào work/e0.jsonl      # mỗi dòng một ảnh
```

Chấm:

```bash
python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json
```

> `llm_int8_skip_modules=["visual","lm_head"]` giữ ViT (kể cả projector) ở fp16. Nếu ViT bị lượng tử hoá 4-bit, trọng số của merger là trọng số lượng tử và **không thể huấn luyện đầy đủ** ở E2/E3.

**Kiểm tra:** `parse_valid_rate` là bao nhiêu? In 5 output thô để xem model trả gì (có bọc ```json không? có chữ thừa không?).

**Câu hỏi:** hai lỗi phổ biến nhất của E0 là gì (sai thang, bịa box, sai lớp, JSON hỏng)? Kèm một ví dụ output.

## Bước 4 · Ví dụ huấn luyện và mask loss

```
hàm encode_example(image, prompt, answer):
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True,
                     tokenize=True, return_dict=True, return_tensors="pt")
    full_enc   = processor.apply_chat_template([user, assistant(answer)], add_generation_prompt=False, ...)
    n = độ dài prompt_enc.input_ids
    KIỂM TRA full_enc.input_ids[0, :n] == prompt_enc.input_ids[0]    # prompt là tiền tố của chuỗi đầy đủ
    labels = bản sao của full_enc.input_ids;  labels[:, :n] = -100   # chỉ học phần trả lời
    trả full_enc kèm labels

hàm collate(batch):
    pad input_ids / attention_mask / mm_token_type_ids / labels đến cùng độ dài (labels đệm = -100)
    pixel_values, image_grid_thw: nối (cat) theo chiều 0, KHÔNG pad
```

**Kiểm tra:** với một ví dụ, in số token ảnh (phải là **220** cho ảnh 640×360), và giải mã riêng các token có `labels != -100`: kết quả phải đúng là JSON đáp án (cộng `<|im_end|>`).

**Câu hỏi:** vì sao không tính loss trên prompt và token ảnh? Điều gì xảy ra nếu tính?

## Bước 5 · Setup LoRA / đóng băng (E1, E2, E3)

```
hàm apply_setup(model, setup):
    mọi tham số: requires_grad = False
    model.enable_input_require_grads()            # cần cho gradient checkpointing
    mục tiêu LoRA (regex trên tên module):
        LLM (E1, E2, E3): model.language_model.layers.N.self_attn.{q,k,v,o}_proj
                          model.language_model.layers.N.mlp.{gate,up,down}_proj      # 36 × 7 = 252 module
        ViT (chỉ E3):     model.visual.blocks.N.attn.{qkv,proj}
                          model.visual.blocks.N.mlp.{linear_fc1,linear_fc2}          # 24 × 4 = 96 module
    peft_model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                  target_modules=<regex>))
    nếu setup là E2 hoặc E3:
        với mọi tham số có tên chứa "visual.merger" hoặc "visual.deepstack_merger_list":
            đưa về float32;  requires_grad = True
    in tổng số tham số huấn luyện
```

Gợi ý quy mô (để bạn đối chiếu): LLM + embedding ≈ 4,0B; ViT ≈ 415M; trong ViT, `visual.merger` ≈ 27M và `visual.deepstack_merger_list` ≈ 82M (DeepStack lấy đặc trưng ở lớp 5, 11, 17 của ViT). Tổng tham số Qwen3-VL-4B ≈ 4,44B.

**Kiểm tra:** `len(matched_modules)` phải là 252 (E1, E2) và 348 (E3). Số tham số huấn luyện của E2 phải lớn hơn E1 khoảng 109M.

**Câu hỏi:** vì sao merger phải huấn luyện ở float32 trong khi các trọng số còn lại ở fp16? Vì sao không thể huấn luyện đầy đủ merger nếu ViT bị lượng tử hoá 4-bit?

## Bước 6 · Vòng huấn luyện

```
tối ưu = AdamW(các tham số requires_grad, lr=2e-4, weight_decay=0)
lịch lr: warmup 5% bước đầu, sau đó cosine về 0
scaler = GradScaler("cuda")                                   # fp16
bật gradient checkpointing (use_reentrant=False); model.config.use_cache = False
lặp đến khi đủ max_steps = 300 bước tối ưu:
    lặp 4 lần (tích luỹ gradient), mỗi lần một batch 2 ảnh:
        với autocast(fp16):  loss = model(**batch).loss / 4
        nếu loss không hữu hạn: DỪNG và báo (nguy cơ tràn số fp16)
        scaler.scale(loss).backward()
    scaler.unscale_(opt); clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); opt.zero_grad()
    mỗi 100 bước: lưu adapter  VÀ  lưu riêng các tham số projector đã huấn luyện (E2, E3)
ghi run.json: trainable_params, peak_vram_gb (torch.cuda.max_memory_allocated), train_minutes
```

> `model.save_pretrained()` của peft **chỉ lưu adapter LoRA**. Projector (E2/E3) được huấn luyện đầy đủ, **không nằm trong adapter**: nếu quên lưu riêng, lúc suy luận bạn sẽ nạp lại projector gốc và kết quả E2/E3 trông giống E1.
> Notebook lưu adapter và projector vào `/kaggle/working/day10-output/checkpoints` mỗi 100 bước. Lưu Kaggle Notebook Version sau mỗi setup để giữ output.

**Kiểm tra:** loss giảm đều và hữu hạn; ghi lại VRAM đỉnh và thời gian.

**Câu hỏi:** setup nào dùng nhiều VRAM nhất và vì sao (gợi ý: gradient qua ViT, tham số float32 của projector)?

## Bước 7 · Suy luận và chấm E1–E3

```
nạp model gốc (cùng cấu hình 4-bit như bước 3) → nạp adapter LoRA → nạp lại tham số projector (nếu có)
suy luận trên test như bước 3 → work/e1.jsonl
```

```bash
python scripts/score.py --pred work/e1.jsonl --name E1 --out report/evidence/e1.json \
    --meta trainable_params=<số> --meta peak_vram_gb=<số> --meta train_minutes=<số>
```

Mẹo: gỡ lỗi trên 40 ảnh của `val` (`--split val`); chỉ chấm `test` một lần cho mỗi setup.

**Kiểm tra:** E1 phải hơn hẳn E0. Nếu không, kiểm tra mask loss (bước 4) và thang toạ độ (bước 2).

**Câu hỏi:** lập bảng E0–E3 trong `REPORT.md` (F1, CI95, FP/ảnh, F1 từng lớp, VRAM, thời gian). Setup nào đáng đồng tiền nhất? Chênh lệch có vượt khoảng tin cậy không?

## Bước 8 · E4: ablation (chọn một)

Đổi **đúng một biến** so với setup tốt nhất của E1–E3 (xem `configs/e4.yaml`) và chạy cả hai giá trị (một trong hai có thể chính là setup gốc):

| Biến | Cách đổi |
|---|---|
| `rank` | r = 4 và r = 64 (đặt alpha = 2·r) |
| `data_size` | 10% và 100% ảnh train |
| `coord_scale` | `norm1000` và `pixel`: đổi **cả** nhãn, prompt, và chấm với `--coord pixel` |

Lưu: `report/evidence/e4_<biến>_<giá trị>.json`.

**Câu hỏi:** kết luận của bạn có đáng tin không khi test chỉ có 311 ảnh? (Nhìn CI95.)

## Bước 9 · E5 (tuỳ chọn, +10 điểm): GRPO với reward IoU

```
hàm reward(completion, gt_boxes):
    pred = parse_output(completion)             # dùng được scripts.score.parse_output
    nếu pred là None: trả 0.0                   # JSON hỏng
    nếu không có gt và không có pred: trả 1.0
    ghép đôi tham lam theo IoU giảm dần, cùng lớp, chỉ nhận IoU ≥ 0.5
    tổng_iou = tổng IoU các cặp khớp
    trả 0.2 + 0.8 * tổng_iou / max(số_gt, số_pred, 1)    # box bịa bị phạt qua mẫu số
```

Viết **test cho reward trước khi chạy GRPO** (hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng). Hàm reward là Python thuần nên test được mà không cần GPU.

GRPO: khởi tạo từ adapter tốt nhất, `num_generations=4`, `max_completion_length=128`, khoảng 60 bước, `use_vllm=False`. Nếu làm E5, cài `trl==1.9.2` và dùng `trl.GRPOTrainer` (hãy đọc docs của đúng phiên bản này).

**Câu hỏi:** có dấu hiệu reward hacking không (model trả `[]` hoặc một box khổng lồ)? So với SFT, F1 thay đổi thế nào và có vượt CI95 không?

## Khi gặp sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `CUDA out of memory` | batch quá lớn, quên gradient checkpointing | `batch_size=1`, `grad_accum=8`, bật gradient checkpointing |
| Loss `nan`/`inf` | tràn số fp16 | giảm lr; thử `batch_size=1`; kiểm tra bạn đã mask đúng |
| F1 E2/E3 ≈ E1 chính xác | quên nạp lại projector lúc suy luận | lưu và nạp `extra_trainable` (bước 6) |
| `parse_valid_rate` thấp | model lặp vô hạn / bị cắt | tăng `max_new_tokens`, kiểm tra token kết thúc `<|im_end|>` trong nhãn |
| F1 ≈ 0 mà JSON hợp lệ | sai thang toạ độ (pixel thay vì 0–1000) | xem lại bước 2 và `--coord` |
| Mất session giữa chừng | Kaggle session kết thúc | lấy checkpoint từ Notebook Version đã lưu hoặc chạy lại setup còn thiếu |

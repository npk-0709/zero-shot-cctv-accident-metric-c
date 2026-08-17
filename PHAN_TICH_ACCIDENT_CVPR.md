# Phân tích `accident-cvpr.ipynb`

> Notebook gốc nguyên khối (103 cell, 2.79 MB, Python 3.12) chạy end-to-end từ setup môi
> trường đến sinh file submission. Đây là **tổ tiên** mà `core/shared_pipeline.ipynb` và
> `legacy/classical_pipeline.ipynb` được tách ra từ đó.
>
> Tài liệu này phân tích nội dung, đối chiếu với `core/`, liệt kê tài nguyên còn thiếu, và
> hướng dẫn chạy. Xem thêm [`HUONG_DAN_03_SPATIAL_S.md`](HUONG_DAN_03_SPATIAL_S.md) cho
> notebook metric S.

---

## TL;DR — 6 điều quan trọng nhất

1. **File này cũ hơn `core/` nhưng lại chạy được tốt hơn ở một điểm quyết định.**
   `run_inference_vlm` ở đây (cell 76) sạch — chỉ gọi 3 stage. Bản trong core gọi thêm
   `predict_accident_time_ensemble`, một hàm **chưa bao giờ được viết** ở bất kỳ đâu trong
   repo. Đó là lỗi hồi quy do bước modular hoá, và nó khiến `run_inference_vlm` của core
   chắc chắn crash.
2. **Nó là notebook DUY NHẤT có Section 8 (Test Inference & Submission).** Không có phần này,
   `core/` không thể tạo file nộp. Section 8 có switch dry-run `SUBSET_N`, checkpoint mỗi 25
   video để chạy xuyên nhiều session, ghép schema với `sample_submission.csv`, và một check
   chống "submission suy biến".
3. **Nó chứa 2 trong 3 thứ đang chặn `notebooks/03_spatial_S.ipynb`**: `const_eval` (cell 83)
   và cách dựng DataFrame đánh giá (cell 86–87, 93). Cell 88/89 của nó chính là cell 4/5 của
   notebook 03.
4. **Nó dừng giữa đường vì OOM.** Cell 93 chết ở video thứ 8/20. Nguyên nhân là
   `device_map='auto'` ở cell 72 — core **đã sửa** lỗi này, còn file này thì chưa. Phải port
   bản sửa sang trước khi chạy.
5. **Mọi pipeline đo được trong file này đều THUA constant floor**: floor 0.2446, tracking
   0.2377, classical baseline 0.2205. Chỉ riêng thành phần C vượt được hằng số của nó.
6. **Section 8 chưa bao giờ được chạy** (`execution_count = None` ở cell 96–100). Chưa từng
   có file submission nào được sinh ra từ notebook này.

---

## Mục lục

- [Phần 1 — File này là gì, và quan hệ với repo](#phần-1--file-này-là-gì-và-quan-hệ-với-repo)
- [Phần 2 — Bản đồ 103 cell](#phần-2--bản-đồ-103-cell)
- [Phần 3 — Số đo thật đã lưu trong file](#phần-3--số-đo-thật-đã-lưu-trong-file)
- [Phần 4 — Trạng thái chạy và sự cố OOM](#phần-4--trạng-thái-chạy-và-sự-cố-oom)
- [Phần 5 — Tài nguyên còn thiếu](#phần-5--tài-nguyên-còn-thiếu)
- [Phần 6 — Vá lỗi bắt buộc trước khi chạy](#phần-6--vá-lỗi-bắt-buộc-trước-khi-chạy)
- [Phần 7 — Hướng dẫn chạy từng bước](#phần-7--hướng-dẫn-chạy-từng-bước)
- [Phần 8 — Cái gì nên trích ra cho 3 notebook con](#phần-8--cái-gì-nên-trích-ra-cho-3-notebook-con)
- [Phần 9 — Xử lý sự cố](#phần-9--xử-lý-sự-cố)
- [Phụ lục — Bảng tra cứu cell](#phụ-lục--bảng-tra-cứu-cell)

---

## Phần 1 — File này là gì, và quan hệ với repo

### 1.1 Vị trí trong repo

```text
zero-shot-cctv-accident/
├── accident-cvpr.ipynb          ← FILE NÀY. 103 cell, 2.79 MB. Chưa được git track.
├── core/
│   ├── shared_pipeline.ipynb    ← 69 cell. Tách ra từ file này, ĐÃ sửa OOM, nhưng
│   │                              làm hỏng run_inference_vlm và mất Section 8.
│   └── shared_pipeline.py       ← export .py của core
├── legacy/
│   └── classical_pipeline.ipynb ← nhánh tracking-only, cũng tách ra từ file này
└── notebooks/
    ├── 01_temporal_T.ipynb      ← "vỏ" %run core, override Stage 1/2
    ├── 02_type_C.ipynb          ← "vỏ" %run core, override cascade loại va chạm
    └── 03_spatial_S.ipynb       ← "vỏ" %run core, override Stage 3
```

### 1.2 Bằng chứng file này cũ hơn `core/`

Tôi đối chiếu định nghĩa hàm giữa hai file. Kết quả có hướng rõ ràng:

**Core có, file này KHÔNG có** (7 hàm — đều là bản sửa lỗi/nâng cấp thêm về sau):

| Hàm | Core cell | Mục đích |
|---|---|---|
| `vlm_generate_oom_safe` | 53 | Retry khi OOM bằng cách giảm nửa số frame. **Chính là bản vá cho sự cố ở cell 93 của file này.** |
| `classify_type_cascade` | 62 | Bỏ phiếu 3 prompt + cổng entropy cho loại va chạm |
| `_vote_entropy` | 62 | Phụ trợ cho cascade |
| `_torch_smoke_test` | 8 | Kiểm tra CUDA thật bằng subprocess (file này chỉ kiểm tra hời) |
| `_diagnose_arch_mismatch` | 8 | Chẩn đoán compute capability vs `torch.cuda.get_arch_list()` |
| `_looks_like_dataset_root` | 6 | Dò `BASE_DIR` bền hơn (file này dò đơn giản hơn) |
| `_sh` | 8 | Wrapper subprocess |

Thêm bằng chứng mạnh nhất — cell 72 của file này vẫn dùng:

```python
# device_map='auto', not {'': DEVICE}: Kaggle offers a T4 x2 accelerator, and
# pinning everything to cuda:0 throws away the second 16 GB. 'auto' shards
# across whatever is present and is identical on a single GPU.
vlm_model = AutoModelForImageTextToText.from_pretrained(
    VLM_CHECKPOINT, quantization_config=_bnb, device_map='auto',
    dtype=torch.float16, trust_remote_code=True).eval()
```

Trong khi core cell 52 đã thay bằng cách ghim cả model vào một GPU trống nhất, kèm comment
giải thích chính xác vì sao `'auto'` gây OOM. Nói cách khác: **core là bản sau khi sự cố ở
cell 93 của file này được chẩn đoán.**

### 1.3 Nhưng core lại làm hỏng hai thứ

**Thứ nhất — `run_inference_vlm`.** Bản ở đây (cell 76) gọi đúng 3 stage:

```python
pred    = stage1_full_scan(video_path, duration, scene)
t_final = stage2_time_refine(video_path, pred['accident_time'], duration)
pt      = stage3_grounding(video_path, t_final)
```

Bản trong core cell 63 chèn thêm một khối "classical anchor" giữa Stage 1 và Stage 2:

```python
TEMPORAL_ANCHOR_DELTA_MAX = 3.0
...
t_classical = predict_accident_time_ensemble(video_path)   # ← hàm KHÔNG TỒN TẠI
correction = float(np.clip(t_vlm - t_classical, -TEMPORAL_ANCHOR_DELTA_MAX,
                           TEMPORAL_ANCHOR_DELTA_MAX))
pred['accident_time'] = float(np.clip(t_classical + correction, 0.0, duration))
```

`predict_accident_time_ensemble` không có trong core, không có trong file này, không có trong
`legacy/`. Hàm gần nhất về ý tưởng là `predict_accident_time` (cell 54) và
`predict_accident_time_refined` (cell 55) của file này. Ai viết khối anchor đó đã đặt tên cho
một hàm dự định viết rồi không viết.

**Thứ hai — Section 8 bị mất hoàn toàn.** Core kết thúc ở cell 68 (dựng tập calibration 20
video). Không có inference trên tập test, không có sinh submission, không có checkpoint.

### 1.4 Bốn hàm classical mà core gọi nhưng không định nghĩa — đều nằm ở đây

Đây là lý do thứ hai để giữ file này: nó là **nguồn duy nhất** cho các hàm sau.

| Hàm | Cell | Core dùng ở đâu | Nội dung |
|---|---|---|---|
| `predict_accident_time` | 54 | `run_inference_baseline` (core cell 65) | Neo thời gian từ z-score của frame-diff |
| `predict_collision_type_clip` | 57 | `run_inference_baseline` + `_type_fallback` | Phân loại zero-shot bằng CLIP |
| `predict_impact_location_flow` | 61 | `run_inference_baseline` | Centroid có trọng số theo độ lớn optical flow |
| `clip_type_probabilities` | 57 | (dùng nội bộ ở cell 85) | Trả về dict xác suất 5 loại |
| `predict_accident_time_refined` | 55 | — | Tinh chỉnh bằng Perception Encoder (kết quả âm) |
| `predict_impact_location_grounded` | 62 | — | Grounding bằng OWLv2 (kết quả âm, S=0.156) |

Nếu bạn muốn `run_inference_baseline` của core chạy được, đây là chỗ để copy 3 hàm đầu.

---

## Phần 2 — Bản đồ 103 cell

73 cell code + 30 cell markdown. Cấu trúc theo chuẩn CVPR 10 section.

| Cell | Section | Nội dung | Thời gian chạy |
|---|---|---|---|
| 0–2 | — | Tiêu đề, Introduction, Table of Contents | — |
| 3–8 | **0. Environment Setup** | Import, palette, `SEED=42`, dò `BASE_DIR`, audit đường dẫn, `pip_install` | 3–6 phút |
| 9–13 | **1. Data Acquisition** | `labels.csv` (2211 dòng), `test_metadata.csv` (2027 dòng), `sample_sub`, `real_videos`, `synthetic_videos` | vài giây |
| 14–19 | **2. Data Inspection** | Kiểu dữ liệu, missing, `extract_video_meta` trên 20 clip mẫu | ~30 giây |
| 20–23 | **3. Data Cleaning** | Dedup, chuẩn hoá nhãn `type`, clamp toạ độ về [0,1], resolve + kiểm tra tồn tại đường dẫn video | ~30 giây |
| 24–32 | **4. EDA** | Phân bố loại, histogram thời gian, scatter/KDE không gian, kích thước bbox, ma trận tương quan, `sample_frames` | 1–3 phút |
| 33–39 | **5. Feature Engineering** | `compute_frame_diff_series`, `score_temporal_anomaly`, `compute_flow_magnitude_map`, `extract_frames_window`, `COLLISION_PROMPTS` | 1–2 phút |
| 40–47 | **6.1 Model Loading** | `LOAD_LEGACY_MODELS=False`, YOLOv8s, CLIP ViT-B/32, 3 cell skip legacy (PE/Qwen2.5-VL/OWLv2), VRAM audit | 1–2 phút |
| 48–50 | **6.2 Tracking** | `iou_xyxy`, `box_gap`, `SortLiteTracker` (Hungarian + constant velocity), `extract_vehicle_tracks`, `track_kinematics` (Savitzky-Golay) | — |
| 51–52 | **6.3 Collision Analysis** | `_pair_contact`, `_impact_time`, `_impact_point`, `_type_from_geometry`, `FEATURE_COLS`, `analyze_video_collision` | — |
| 53–55 | **6.4 Stage 1 — When** | `predict_accident_time`, `_pe_accident_scores`, `predict_accident_time_refined` | — |
| 56–59 | **6.5 Stage 2 — What** | `clip_type_probabilities`, `predict_collision_type_clip`, `_qwen_generate`, `qwen_type_votes`, `qwen_scene_context`, `predict_collision_type_final` | — |
| 60–62 | **6.6 Stage 3 — Where** | `predict_impact_location_flow`, `predict_impact_location_grounded` | — |
| 63–64 | **6.7 Constant Prior** | `SIGMA_T_LIST=(0.5,1.0,2.0)`, `SIGMA_X=0.0952`, `SIGMA_Y=0.1353`, `fit_constant_predictor`, `CONST` | ~10 giây |
| 65–66 | **6.8 Sim-to-Real** | `_probe` — tracking có bắt được gì trên CCTV thật không | — |
| 67–70 | **6.9 Supervised Classifier** | HistGradientBoosting trên 13 feature kinematics — **KẾT QUẢ ÂM**, `_Tm`/`_Sm`, `TIME_OFFSET`, `TYPE_CLASSIFIER` | — |
| 71–76 | **6.10 Qwen3-VL 3-stage** | Load Qwen3-VL-8B 4-bit + `VLM_CFG`, prompts + `sample_frames_stamped` + `vlm_generate`, **canary**, 3 stage function, `run_inference_vlm` | 8–15 phút (tải model) |
| 77–78 | **6.11 Assembled** | `run_inference_baseline`, `run_inference_final`, `TRACK_EVIDENCE_MIN` | — |
| 79–81 | **7. Evaluation** | `temporal_score`/`spatial_score`/`classification_score`/`accident_score`/`score_predictions`, dựng `diverse_videos` (20 video) | ~10 giây |
| 82–83 | **7a. Constant Floor** | `const_eval` — **đây là thứ `03_spatial_S.ipynb` đang thiếu** | ~5 giây |
| 84–85 | **7b. Per-Signal Ablation** | Từng tín hiệu đo riêng vs hằng số của nó + sweep `TRACK_EVIDENCE_MIN` | ~10 phút |
| 86–89 | **7. Comparison** | Chạy baseline + final trên 20 video, `eval_baseline`/`eval_final`, histogram, đường cong độ nhạy | ~15 phút |
| 90–91 | **7d-bis. Raw VLM dump** | In output THÔ của Stage 1, không parse — cell chẩn đoán quan trọng nhất | ~4 phút |
| 92–93 | **7d. Qwen3-VL trên calibration** | `eval_vlm` — **cell OOM, chưa chạy xong** | 30–50 phút (nếu sửa OOM) |
| 94 | **7c. Negative Result** | Phân loại bằng góc optical flow — tài liệu hoá để không thử lại | — |
| 95–100 | **8. Submission** | `SUBSET_N`, checkpoint, ghép schema, 3 biểu đồ phân bố dự đoán | 17–22 giờ (full run) |
| 101–102 | **9–10** | Conclusion (rất đáng đọc), References | — |

### 2.1 Ba cell chẩn đoán đáng giá nhất trong file

Nếu bạn chỉ có thời gian đọc 3 cell, hãy đọc 3 cell này:

**Cell 91 — dump output thô của VLM.** Không parse, không coerce. Nó tồn tại vì lần chạy đầu
tiên dự đoán `t-bone` cho toàn bộ 7 clip, và có 2 giả thuyết cần phân biệt: (a) model trả lời
tệ, (b) model trả lời tốt nhưng ta parse sai (JSON trong code fence, nhãn `"T-bone"` hoặc
`"t_bone"`, bị cắt ở `max_new_tokens`). Markdown cell 90 nói thẳng: *"Guessing between these
is pointless when the raw string is one print away."*

**Cell 85 — ablation từng tín hiệu.** Mỗi tín hiệu được đo trên độ phủ của nó, rồi trộn với
hằng số ở những chỗ nó không bắt được, và in verdict `KEEP`/`DELETE`. Đây là phép đo mà
pipeline trước đó thiếu — không có nó thì không biết trong 5 model cái nào đang gánh điểm và
cái nào đang phá.

**Cell 97 — check submission suy biến.** Sau khi ghi file, nó đếm số loại phân biệt và std của
toạ độ. Lý do: *"A pipeline that has silently stopped working produces a submission with almost
no variety — one type, one point, one time. That is indistinguishable from a valid file until
the leaderboard says nothing moved."*

---

## Phần 3 — Số đo thật đã lưu trong file

Toàn bộ số dưới đây là output **đã lưu trong notebook**, không phải suy đoán. Đo trên tập
calibration 20 video synthetic (4 video mỗi loại, `random_state=42`).

### 3.1 Bảng so sánh chính (cell 87)

| | T | S | C | ACCIDENT score |
|---|---|---|---|---|
| **Constant floor** (cell 83) | **0.3800** | **0.2159** | 0.2000 | **0.2446** |
| Baseline classical | 0.1890 | 0.2007 | 0.3000 | 0.2205 |
| Final tracking | 0.2905 | 0.1497 | **0.4000** | 0.2377 |

Verdict do chính notebook in ra:

```
T baseline: 0.1890  LOSES TO constant 0.3800
T final   : 0.2905  LOSES TO constant 0.3800
S baseline: 0.2007  LOSES TO constant 0.2159
S final   : 0.1497  LOSES TO constant 0.2159
C baseline: 0.3000  BEATS    constant 0.2000
C final   : 0.4000  BEATS    constant 0.2000
```

**Cả hai pipeline đều thua hằng số về điểm tổng.** Năm model và khoảng 11,5 giờ T4 cho ra kết
quả tệ hơn việc đoán ba con số cố định. Chỉ có thành phần C vượt được — và ngay cả điều đó cũng
có dấu hoa thị: tập calibration cân bằng 4-mỗi-loại nên C = 0.2 là mức ngẫu nhiên, còn dưới
prior tự nhiên của tập synthetic thì cùng hằng số `rear-end` đó đạt **0.3591** — cao hơn mọi
bộ phân loại zero-shot đã thử.

### 3.2 Phân rã C theo loại (final pipeline, cell 87)

| Ground truth | Accuracy | n |
|---|---|---|
| head-on | 0.75 | 4 |
| rear-end | 0.50 | 4 |
| sideswipe | 0.25 | 4 |
| single | 0.25 | 4 |
| t-bone | 0.25 | 4 |

Ma trận nhầm lẫn cho thấy `single` bị nhận thành `head-on` 2/4 lần, và `sideswipe`/`t-bone`
tán ra khắp các lớp — dấu hiệu tín hiệu hình học không phân biệt được đúng những lớp nó được
thiết kế để phân biệt.

### 3.3 Ablation từng tín hiệu (cell 85)

Tracking chỉ tìm được va chạm ở **70% video** (14/20). Ba loại `sideswipe` và hai `t-bone`
hoàn toàn không bắt được (`evidence = 0.0000`).

| Thành phần | Tín hiệu | Độ phủ | Điểm trên độ phủ | Trộn với hằng số |
|---|---|---|---|---|
| T | constant | 1.00 | 0.6031 | 0.6031 |
| T | frame-diff anchor | 1.00 | 0.3141 | 0.3141 |
| T | tracking | 0.70 | 0.3483 | 0.4578 |
| S | constant | 1.00 | 0.1607 | 0.1607 |
| S | optical-flow centroid | 1.00 | 0.1177 | 0.1177 |
| S | tracking | 0.70 | 0.1547 | 0.1326 |
| C | constant | 1.00 | 0.2000 | 0.2000 |
| C | CLIP zero-shot | 1.00 | 0.3000 | — |
| C | track geometry | 0.45 | 0.3333 | — |

> ### ⚠ Một sai sót thật trong cell 85 mà bạn cần biết
>
> Cell 85 tự định nghĩa lại hàm điểm bằng **sigma vô hướng đã bị khai tử**, không dùng hàm
> điểm chính thức ở cell 80:
>
> ```python
> def sc_T(pred): return np.exp(-0.5 * ((pred - gt_t) / SIGMA_T) ** 2)          # SIGMA_T = 2.0
> def sc_S(px, py): return np.exp(-0.5 * ((px-gt_x)**2 + (py-gt_y)**2) / SIGMA_S**2)  # SIGMA_S = 0.1
> ```
>
> Trong khi cell 64 nói rõ `SIGMA_T`/`SIGMA_S` chỉ *"kept only so the constant grid-search
> below has a scalar to optimise against; the scoring functions in Section 7 use the real
> definition above."*
>
> Hệ quả cụ thể: hằng số T trong bảng ablation là **0.6031** (chỉ σ=2.0, mức rộng rãi nhất)
> nhưng ở Section 7a là **0.3800** (trung bình 3 mức σ = 0.5/1.0/2.0). Hằng số S là **0.1607**
> (đẳng hướng σ=0.1) nhưng ở 7a là **0.2159** (bất đẳng hướng 0.0952/0.1353).
>
> Nghĩa là **các verdict `KEEP`/`DELETE` của cell 85 có thể đảo chiều** dưới hàm điểm đúng.
> Nếu bạn dựa vào bảng ablation để quyết định giữ hay bỏ tín hiệu nào, hãy sửa `sc_T`/`sc_S`
> thành `temporal_score`/`spatial_score` của cell 80 rồi chạy lại. Đây là việc rẻ (không cần
> GPU, chỉ tính lại từ `diag` đã có).

### 3.4 Đo trên pipeline Qwen3-VL (cell 93) — chưa hoàn thành

Cell chết ở video 8/20. Bảy video kịp chạy:

```
1/20  Town03_head-on_wet_48     t= 3.67s  (0.43, 0.50)  t-bone   77.1 s/video
2/20  Town06_head-on_wet_06     t=14.47s  (0.72, 0.28)  t-bone   81.5 s/video
3/20  Town03_head-on_night_40   t= 5.06s  (0.80, 0.69)  t-bone   88.4 s/video
4/20  Town06_head-on_wet_01     t= 6.06s  (0.56, 0.72)  t-bone   87.3 s/video
5/20  Town04_rear-end_sunset_13 t= 5.35s  (0.68, 0.36)  t-bone   86.7 s/video
6/20  Town04_rear-end_rain_09   t= 3.59s  (0.64, 0.66)  t-bone   84.9 s/video
7/20  Town05_rear-end_rain_142  t=13.97s  (0.00, 0.00)  t-bone   85.5 s/video
```

Ba điều đọc được từ 7 dòng này:

1. **Loại va chạm collapse hoàn toàn về `t-bone`** — 4 cái đúng ra là `head-on`, 3 cái là
   `rear-end`, tức **C = 0/7**. Đây chính là hiện tượng về sau sinh ra toàn bộ cuộc điều tra
   bias trong `02_type_C` (đo được prior content-free: `t-bone` 0.48 so với `head-on` 0.04).
2. **Thời gian và toạ độ thì biến thiên theo từng clip**, nên model *có* phản ứng với nội dung
   video — chỉ riêng nhãn loại bị sập. Đây là phân biệt quan trọng: không phải model chết.
3. **Video 7 trả về `(0.00, 0.00)`** — góc trên trái. Đó là dấu hiệu Stage 3 parse ra rác hoặc
   trả về giá trị suy biến, đúng loại lỗi mà notebook `03_spatial_S` cần điều tra.

Mỗi video đều OOM 2 lần trước khi thành công (32 frame → 24 → 12). Nghĩa là Stage 1 trả lời từ
12 frame trải trên ~20 giây, tức khoảng 1 frame mỗi 1,7 giây — **hoàn toàn có thể bỏ sót thời
điểm va chạm**. Con số 77–88 s/video đã bao gồm thời gian OOM retry.

### 3.5 Mốc tham chiếu ngoài (từ markdown cell 92)

| Hệ thống | Điểm trên tập test thật |
|---|---|
| Constant floor | ~0.24 |
| Pipeline tracking của repo này | ~0.24 |
| **Một lần gọi VLM đơn lẻ** (đội hạng 1, arXiv:2605.29325) | **0.42238** |
| Hệ 3-stage 32B thắng giải | **0.57080** |

Khoảng cách giữa 0.24 và 0.42 nói lên điều quan trọng nhất: **một lần gọi Qwen3-VL đúng cách
gần như gấp đôi toàn bộ pipeline 5 model + tracker của notebook này.** Đó là lý do Section
6.10 tồn tại và các Section 6.2–6.9 được giữ lại chỉ như kết quả âm có tài liệu.

---

## Phần 4 — Trạng thái chạy và sự cố OOM

### 4.1 Notebook đã chạy đến đâu

`execution_count` cao nhất là **67**. Các cell **chưa từng chạy** (hoặc đã bị sửa sau khi chạy):

| Cell | Nội dung | Ý nghĩa |
|---|---|---|
| 72 | Load Qwen3-VL + `VLM_CFG` | Đã được **sửa** sau lần chạy cuối (đổi `{'': DEVICE}` → `'auto'`) nhưng chưa chạy lại |
| 75 | 3 stage function | Cũng đã sửa, chưa chạy lại |
| 91 | Dump output thô của VLM | Cell chẩn đoán mới thêm, chưa chạy |
| 96–100 | **Toàn bộ Section 8** | **Chưa bao giờ chạy — chưa từng có submission nào được tạo** |

Đọc theo trình tự: notebook chạy đến cell 93, gặp OOM, người viết chẩn đoán nguyên nhân, sửa
cell 72/75, thêm cell chẩn đoán 91 — rồi **dừng**, và bản sửa hoàn chỉnh về sau đi vào
`core/shared_pipeline.ipynb`. File này là ảnh chụp đúng thời điểm đó.

### 4.2 Lỗi OOM: nguyên nhân gốc

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 938.00 MiB.
GPU 1 has a total capacity of 14.56 GiB of which 598.81 MiB is free.
Of the allocated memory 12.80 GiB is allocated by PyTorch,
and 1.03 GiB is reserved by PyTorch but unallocated.
```

Chuỗi nhân quả, theo đúng comment mà core để lại:

1. Cell 72 dùng `device_map='auto'`. Nó chia model qua 2 GPU T4 dựa trên **trọng số tĩnh**.
2. Nhưng `'auto'` **không có cách nào biết** rằng KV-cache lúc decode sẽ phình lên tỉ lệ với số
   visual token (`whole_limit × image_max_side`), và nó sẽ nằm trên GPU nào giữ các layer
   decoder cuối.
3. Lần chạy này nó dồn phần lớn ~6,8 GB weight **và** KV-cache đang phình lên vào GPU 1.
4. GPU 1 hết chỗ cho một cấp phát 938 MiB.

Đáng chú ý: **lỗi này không phải do frame budget quá lớn.** `VLM_CFG` ở đây đã là bản đã được
suy nghĩ kỹ (`whole_limit=32`, `image_max_side=448` — khoảng 4.608 visual token, đúng bằng
core). Vấn đề là phân bổ giữa 2 GPU, không phải tổng dung lượng.

### 4.3 Bản sửa (đã có trong core, phải port sang)

Core cell 52 thay `'auto'` bằng: ghim **cả model** vào GPU đang trống nhất. Lý do 6,8 GB thừa
sức nằm gọn trên một T4 15,6 GB, và như vậy không còn chuyện mất cân bằng giữa 2 GPU phải lý
giải. Cách này cũng không hardcode "ưu tiên GPU 1" (sẽ vỡ trên máy 1 GPU).

Cộng thêm `vlm_generate_oom_safe` (core cell 53) để Stage 2/Stage 3 cũng có retry — ở file này
chỉ `stage1_full_scan` có vòng retry riêng, còn Stage 2/3 gọi `vlm_generate` trực tiếp, nên một
clip quá khổ có thể giết cả lượt chạy nhiều giờ đã checkpoint.

Code cụ thể ở [Phần 6](#phần-6--vá-lỗi-bắt-buộc-trước-khi-chạy).

---

## Phần 5 — Tài nguyên còn thiếu

### 5.1 Thiếu bắt buộc — không có thì không chạy được

| Tài nguyên | Chi tiết | Lấy ở đâu |
|---|---|---|
| **Dataset `accident`** | 2211 clip synthetic CARLA + 2027 clip CCTV thật, `labels.csv` (19 cột), `test_metadata.csv` (10 cột), `sample_submission.csv`, `annotation_classes.yaml` | Kaggle → Join Competition → Add Input → Competition. **Không nằm trong repo** |
| **GPU ≥ 16 GB VRAM** | Qwen3-VL-8B 4-bit: ~6,9 GB weight + activation cho 32 frame @448px | Kaggle Accelerator `GPU T4 x2` (khuyến nghị) hoặc `P100` |
| **Internet** | `pip install`, `git clone`, tải model HuggingFace | Kaggle Settings → Internet On (cần tài khoản đã xác thực SĐT) |
| **Qwen3-VL-8B-Instruct** | ~16 GB tải về, tự động qua `from_pretrained` | HuggingFace. Nên có `HF_TOKEN` để tránh rate limit |
| **YOLOv8s** | 21,5 MB, tự tải | ultralytics assets |
| **CLIP ViT-B/32** | 338 MB, tự tải | `git+https://github.com/openai/CLIP.git` |

Đường dẫn dataset mà notebook mong đợi:

```text
/kaggle/input/competitions/accident/          (hoặc /kaggle/input/accident/)
├── sim_dataset/
│   ├── labels.csv                            rgb_path, accident_time, accident_frame,
│   │                                         center_x, center_y, x1, y1, x2, y2, type, ...
│   ├── annotation_classes.yaml
│   ├── videos/{head-on,rear-end,sideswipe,single,t-bone}/*.mp4
│   └── video_annotations/
├── videos/                                   2027 clip CCTV thật (phẳng, .mp4)
├── test_metadata.csv                         có cột scene_layout
└── sample_submission.csv                     path, accident_time, center_x, center_y, type
```

### 5.2 Thiếu về code — phải tự vá

| Thiếu | Mức độ | Cách xử lý |
|---|---|---|
| Bản sửa OOM (ghim GPU + `vlm_generate_oom_safe`) | **Chặn cứng** | Port từ core cell 52/53 — [Phần 6.1](#61-patch-1--sửa-oom-bắt-buộc) |
| `classify_type_cascade` | Nên có | Chỉ có trong core cell 62. Không có nó thì loại va chạm lấy thô từ Stage 1 → chính là chỗ collapse về `t-bone` |
| `sc_T`/`sc_S` dùng sigma sai ở cell 85 | Nên sửa | [Phần 6.3](#63-patch-3--sửa-hàm-điểm-của-ablation-nên-làm) |
| `predict_accident_time_ensemble` | Không cần ở đây | Chỉ core cần. **Đừng port khối anchor của core sang file này** |

### 5.3 Thiếu về hạn mức — đây là ràng buộc đáng lo nhất

Cell 93 đo được **77–88 s/video**, đã bao gồm 2 lần OOM retry mỗi video. Sau khi sửa OOM, ước
tính còn khoảng **30–40 s/video** (bỏ được 2 lượt Stage 1 thất bại).

| | s/video | 2027 clip | Số session Kaggle 12h |
|---|---|---|---|
| Như hiện trạng (có OOM retry) | ~82 | **~46 giờ** | 4+ |
| Sau khi sửa OOM (ước tính) | ~35 | **~20 giờ** | 2 |

Kaggle cho **30 giờ GPU mỗi tuần**, mỗi session tối đa 12 giờ. Nghĩa là **một lần nộp đầy đủ
ăn hết khoảng 2/3 quota tuần**. Cơ chế checkpoint ở cell 96 (`preds_checkpoint.csv`, ghi mỗi 25
video) chính là để chia việc này qua nhiều session — đừng vô hiệu hoá nó.

Chính cell 96 cũng tự cảnh báo:

```python
if avg_t * len(real_videos) > 11 * 3600:
    print('[WARNING] Full-set projection exceeds one 12 h Kaggle session -- '
          'the checkpoint lets you split the run across sessions safely')
```

### 5.4 Một khoản lãng phí có thể bỏ ngay

Cell 8 vẫn `git clone` + `pip install -e perception_models` (repo của facebookresearch), dù cell
44 đã `LOAD_LEGACY_MODELS = False` nên Perception Encoder **không bao giờ được dùng**. Comment
out 2 dòng đó tiết kiệm vài phút mỗi lần chạy:

```python
# result = subprocess.run(
#     ['git', 'clone', 'https://github.com/facebookresearch/perception_models.git'], ...)
# pip_install(['-e', 'perception_models'], 'perception_models')
```

---

## Phần 6 — Vá lỗi bắt buộc trước khi chạy

### 6.1 Patch 1 — sửa OOM (bắt buộc)

**Thay phần `from_pretrained` trong cell 72** bằng đoạn dưới. Nó ghim cả model vào GPU trống
nhất thay vì để `'auto'` chia trọng số rồi dồn KV-cache vào một GPU:

```python
    _bnb = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            if DEVICE == 'cuda' else None)
    vlm_processor = AutoProcessor.from_pretrained(VLM_CHECKPOINT)

    # device_map='auto' can bang TRONG SO TINH, nhung khong biet KV-cache luc decode
    # (ti le voi whole_limit x image_max_side) se roi vao GPU nao giu cac layer
    # decoder cuoi. Lan chay truoc no don ~6.8 GB weight VA KV-cache dang phinh vao
    # GPU 1, roi OOM o mot cap phat 938 MiB. Ghim ca model vao GPU trong nhat:
    # 6.8 GB nam gon tren mot T4 15.6 GB, khong con mat can bang nao phai ly giai,
    # va khong hardcode 'uu tien GPU 1' (se vo tren may 1 GPU).
    if torch.cuda.is_available():
        _n_gpu = torch.cuda.device_count()
        _free_by_gpu = [torch.cuda.mem_get_info(g)[0] for g in range(_n_gpu)]
        _best_gpu = int(max(range(_n_gpu), key=lambda g: _free_by_gpu[g]))
        _vlm_device_map = {'': f'cuda:{_best_gpu}'}
        print(f'[STATUS] GPU free memory: '
              f'{[f"{f/1e9:.2f} GB" for f in _free_by_gpu]} -> pinning VLM to cuda:{_best_gpu}')
    else:
        _vlm_device_map = {'': 'cpu'}

    vlm_model = AutoModelForImageTextToText.from_pretrained(
        VLM_CHECKPOINT, quantization_config=_bnb, device_map=_vlm_device_map,
        dtype=torch.float16, trust_remote_code=True).eval()
    print(f'[SUCCESS] {VLM_CHECKPOINT} loaded')

    _n4bit = sum(1 for m in vlm_model.modules() if 'Linear4bit' in type(m).__name__)
    print(f'[STATUS] Linear4bit layers: {_n4bit}  (0 means quantization did NOT apply)')
```

Đồng thời, ở **đầu** khối `try` của cell 72, thêm đoạn giải phóng handle cũ để cell có thể chạy
lại mà không cần restart kernel:

```python
    # Mot lan load that bai/do dang co the de lai tensor con song trong namespace,
    # ma torch.cuda.empty_cache() mot minh khong thu hoi duoc.
    for _name in ('vlm_model', 'vlm_processor'):
        if _name in globals():
            del globals()[_name]
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 6.2 Patch 2 — thêm `vlm_generate_oom_safe` (bắt buộc)

**Thêm vào cuối cell 73.** Ở file này chỉ `stage1_full_scan` có vòng retry riêng; Stage 2,
Stage 3 và các cell chẩn đoán gọi `vlm_generate` trực tiếp, nên một clip quá khổ có thể giết cả
lượt chạy nhiều giờ đã checkpoint.

```python
def vlm_generate_oom_safe(prompt_text, frames, max_new_tokens=256, min_frames=4):
    """vlm_generate co retry OOM: giam nua so frame roi thu lai thay vi crash.

    frames la list (t, PIL.Image) nhu sample_frames_stamped tra ve -- truyen ca
    tuple (khong chi anh) de ham nay giam nhat quan bat ke stage nao goi. Tra ''
    neu min_frames van OOM, de caller co the fallback thay vi chet.
    """
    import gc
    cur = list(frames)
    while True:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            return vlm_generate(prompt_text, [im for _, im in cur], max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if len(cur) <= min_frames:
                print(f'[WARNING] OOM persists at {len(cur)} frames -- giving up, returning ""')
                return ''
            cur = cur[::2] if len(cur) > 2 * min_frames else cur[:min_frames]
            print(f'[WARNING] OOM -- retrying with {len(cur)} frames')
```

Rồi trong **cell 75**, đổi 2 chỗ gọi `vlm_generate` của Stage 2 và Stage 3:

```python
# Trong stage2_time_refine:
j = _extract_json(vlm_generate_oom_safe(TIME_REFINE_PROMPT, merged, 64))

# Trong stage3_grounding:
text = vlm_generate_oom_safe(GROUNDING_PROMPT, frames, 64, min_frames=1)
```

### 6.3 Patch 3 — sửa hàm điểm của ablation (nên làm)

Cell 85 đo bằng sigma vô hướng đã bị khai tử (xem [3.3](#33-ablation-từng-tín-hiệu-cell-85)).
Thay 2 hàm nội bộ để nó dùng đúng hàm điểm chính thức của cell 80:

```python
# [FIX] Dung dung ham diem cua cell 80, khong phai SIGMA_T/SIGMA_S vo huong da
# bi khai tu. Truoc day hang so T bao 0.6031 (chi sigma=2.0) trong khi Section 7a
# bao 0.3800 (trung binh 3 muc sigma), va S bao 0.1607 (dang huong 0.1) trong khi
# 7a bao 0.2159 (bat dang huong). Cac verdict KEEP/DELETE co the dao chieu.
def sc_T(pred):
    return np.array([temporal_score(p, g) for p, g in zip(pred, gt_t)])

def sc_S(px, py):
    return np.array([spatial_score(a, b, gx, gy)
                     for a, b, gx, gy in zip(px, py, gt_x, gt_y)])
```

Việc này **không cần GPU** — chỉ tính lại từ DataFrame `diag` đã có trong bộ nhớ. Nếu bạn đã
chạy cell 85 một lần, chỉ cần chạy lại phần từ `def sc_T` xuống dưới.

### 6.4 Patch 4 — (tuỳ chọn) thêm cascade phân loại từ core

Nếu bạn muốn C tốt hơn, port `classify_type_cascade` + `_vote_entropy` +
`ELIMINATION_PROMPT_TEMPLATE` từ core cell 61–62 vào sau cell 75, rồi trong `run_inference_vlm`
(cell 76) thêm:

```python
    pred['type'] = classify_type_cascade(video_path, t_final, pred['type'])
    pred['type'] = apply_scene_type_postfix(pred['type'], scene)
```

Đánh đổi: thêm 2–4 lần gọi VLM mỗi clip, tức s/video tăng khoảng 40–60%, đẩy full run từ ~20 giờ
lên ~30 giờ. **Hãy đo trên tập calibration trước khi quyết định.** Với hiện tượng collapse về
`t-bone` (C = 0/7), cascade có khả năng đáng giá — nhưng `02_type_C` đã đo được rằng bias nằm ở
prior của model, nên cascade một mình có thể không đủ.

---

## Phần 7 — Hướng dẫn chạy từng bước

### Bước 0 — Đưa file vào git (nó đang untracked)

```powershell
cd C:\Users\Khuong\Desktop\ThuHuyen\zero-shot-cctv-accident
git status --short          # se thay '?? accident-cvpr.ipynb'
```

File 2,79 MB này chưa được commit. Quyết định trước: nếu giữ nó làm tài liệu tham chiếu thì nên
đặt vào `legacy/` cho khớp quy ước repo, vì nó đúng nghĩa là snapshot lịch sử.

### Bước 1 — Cấu hình Kaggle

| Mục | Giá trị | Ghi chú |
|---|---|---|
| **Accelerator** | `GPU T4 x2` | Patch 1 sẽ tự chọn GPU trống nhất. `P100` (1 GPU) cũng chạy được |
| **Internet** | `On` | Bắt buộc. Cần tài khoản đã xác thực SĐT |
| **Persistence** | `Files only` | Để `preds_checkpoint.csv` sống qua nhiều session — **cực kỳ quan trọng** cho full run |
| **Add Input → Competition** | `accident` | Phải Join Competition + đồng ý rules trước |

Tuỳ chọn nhưng nên làm — thêm `HF_TOKEN` vào Add-ons → Secrets, rồi chèn cell này lên **đầu**
notebook (log của lần chạy cũ có cảnh báo `unauthenticated requests to the HF Hub`):

```python
# [SETUP] HF token -- tai Qwen3-VL-8B nhanh hon, tranh rate limit
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
    print('[SETUP] HF_TOKEN da nap')
except Exception as e:
    print(f'[SETUP] Khong nap duoc HF_TOKEN ({e}) -- van chay duoc, chi cham hon')
```

### Bước 2 — Áp Patch 1 và Patch 2 TRƯỚC KHI chạy bất cứ thứ gì

Xem [Phần 6](#phần-6--vá-lỗi-bắt-buộc-trước-khi-chạy). Nếu bỏ qua bước này, cell 93 và cell 96
sẽ OOM y như lần trước, và bạn mất 25–35 phút tải model để phát hiện điều đã biết trước.

### Bước 3 — Chạy tuần tự cell 0 → 74

Chạy **từng cell một** ở lần đầu, đừng Run All. Các mốc phải xác nhận:

| Mốc | Log phải thấy | Nếu sai |
|---|---|---|
| Cell 6 | `BASE_DIR : /kaggle/input/competitions/accident` | Chưa add competition input |
| Cell 7 | Bảng audit **toàn bộ** `exists=True` | Dataset mount sai layout |
| Cell 8 | 5 dòng `install exit code: 0` + `qwen3_vl architecture recognized: True` | Internet chưa bật; nếu `False` thì **Restart Kernel** rồi chạy lại |
| Cell 10 | `synthetic_labels 2211 19` và `test_metadata 2027 10` | `labels.csv` không đọc được |
| Cell 42 | `[SUCCESS] YOLOv8s loaded` | Internet |
| Cell 43 | `[SUCCESS] CLIP ViT-B/32 loaded \| text features for 5 types` | Internet |
| Cell 44–46 | 3 dòng `skipped (LOAD_LEGACY_MODELS=False)` | Nếu nó thực sự load thì bạn mất 4,9 GB VRAM vô ích |
| Cell 63 | `sigma_t = (0.5, 1.0, 2.0) \| sigma_x = 0.0952 sigma_y = 0.1353` | — |
| Cell 64 | `Constants fitted on 2211 synthetic videos: t=6.15s xy=(0.51, 0.51) type=rear-end` | — |
| Cell 72 | `pinning VLM to cuda:N` + `[SUCCESS] ... loaded` + `Linear4bit layers: 368` | `Linear4bit = 0` nghĩa là model đang ở fp16 (~16 GB) → sẽ OOM |
| Cell 72 | `GPU0 ... 15.01 GB free` / `GPU1 ... 8.69 GB free` | < 4 GB free → sẽ OOM |
| Cell 74 | `[CANARY] generation returned: 'OK'` + `PASS -- the model is alive` | **Không đi tiếp nếu fail** |

Thời gian: khoảng **25–35 phút**, phần lớn là tải Qwen3-VL-8B (~16 GB).

> **Canary (cell 74) là chốt an toàn quan trọng nhất.** Nó tồn tại vì đã từng có lần model
> "load thành công" nhưng mọi lần gọi trả chuỗi rỗng, và pipeline lặng lẽ ghi giá trị mặc định
> (`0.35×duration`, `0.5/0.5`, `rear-end`) cho toàn bộ 2027 clip — một hằng số đội lốt submission,
> sinh ra sau 24 phút GPU.

### Bước 4 — Cell 91: dump output thô của VLM (4 phút, làm trước cell 93)

Đây là bước tôi khuyến nghị mạnh nhất. Cell 91 in ra chuỗi thô mà Stage 1 trả về cho 3 clip đầu,
kèm số visual token, danh sách timestamp, kết quả parse, và kết quả sau `normalize_prediction`.
Với hiện tượng collapse về `t-bone` đã ghi nhận, nó phân biệt ngay 2 giả thuyết:

- Nếu output thô thực sự nói `t-bone` → model trả lời sai, cần sửa prompt/frame budget.
- Nếu output thô nói đúng nhưng bị coerce về mặc định → **lỗi parse**, sửa rẻ hơn nhiều.

Cell tự in `>>> JSON DID NOT PARSE -- every field below is a default, not a prediction` khi rơi
vào trường hợp thứ hai.

### Bước 5 — Cell 93: đánh giá Qwen3-VL trên 20 video calibration

**Đây là con số quyết định có nên bỏ 20 giờ cho full run hay không.** Markdown cell 92 nói thẳng:
*"no version of this notebook has ever produced it."*

Cell tự in `s/video` và ETA cho 2027 clip. Ba lưu ý khi đọc kết quả:

- Tập calibration là CARLA, không phải CCTV — nó đo *pipeline*, không đo sim-to-real.
- **Không có scene hint nào kích hoạt ở đây**, vì `scene_layout` chỉ tồn tại cho clip test thật.
  Cả hint trong prompt Stage 1 (+0.00216) và luật t-bone (+0.00466) đều nằm im. Điểm trên tập
  thật đáng lẽ *cao hơn* con số cell này báo.
- n = 20. Một video là 0.05 của C. Đọc khoảng cách, đừng đọc từng chữ số.

Nếu vẫn OOM sau Patch 1: hạ `VLM_CFG['whole_limit']` xuống 16, rồi `image_max_side` xuống 384.
Ghi rõ vào markdown vì điểm đo được sẽ khác mốc gốc.

Thời gian: **30–50 phút** cho 20 video.

### Bước 6 — Cell 96 với `SUBSET_N = 10` (dry run)

```python
SUBSET_N = 10           # 10 clip THAT truoc, do s/video, bat OOM som
SUBMIT_PIPELINE = run_inference_vlm
```

Đây là lần đầu pipeline chạm vào video CCTV **thật**. File ra là
`submission_dryrun_10.csv` — cố tình đặt tên khác để không bao giờ lẫn với bản nộp thật.

Xác nhận: không OOM, `s/video` hợp lý, và toạ độ/thời gian/loại có biến thiên giữa các clip.

### Bước 7 — `SUBSET_N = None` cho full run

```python
SUBSET_N = None         # toan bo 2027 clip
```

Cách chạy xuyên nhiều session:

1. Chạy đến khi Kaggle cắt session ở 12 giờ. Checkpoint đã ghi mỗi 25 video.
2. Session sau: chạy lại cell 0 → 74 (khoảng 30 phút), rồi cell 96. Nó tự phát hiện checkpoint
   và in `[STATUS] Checkpoint found: N videos already processed -- resuming`.
3. Lặp đến khi xong.

Muốn chạy lại sạch thì xoá `preds_checkpoint.csv` trước.

### Bước 8 — Cell 97: sinh submission và kiểm tra suy biến

Cell ghép predictions với `sample_submission.csv` theo cột `path`, điền `CONST` cho dòng không
match, và in:

```
[CHECK] distinct types=N | xy std=X.XXXX | time std=X.XXX
```

**Đừng nộp nếu `distinct types <= 1` hoặc `xy std < 1e-6`.** Đó là dấu hiệu pipeline đã chết và
file bạn có là một hằng số. Cell 98–100 vẽ 3 biểu đồ (phân bố loại, phân bố thời gian, scatter
vị trí) — hãy nhìn cả ba trước khi nộp, đặc biệt scatter: nếu tất cả điểm dồn về một chỗ thì
Stage 3 không trả toạ độ.

---

## Phần 8 — Cái gì nên trích ra cho 3 notebook con

File này là nguồn để vá các blocker của `notebooks/03_spatial_S.ipynb` (và `01_temporal_T.ipynb`,
vốn có cùng lỗi). Chi tiết về các blocker đó nằm trong
[`HUONG_DAN_03_SPATIAL_S.md`](HUONG_DAN_03_SPATIAL_S.md) Phần 4.

### 8.1 `const_eval` — lấy nguyên cell 83

Đây là code gốc, chính xác, cho constant floor. Copy y nguyên:

```python
# [EVAL] Score the constant prior on the calibration split -- the floor
const_df = pd.DataFrame({'path': diverse_labels_df['rgb_path'].to_numpy(),
                         'accident_time': CONST['accident_time'],
                         'center_x': CONST['center_x'],
                         'center_y': CONST['center_y'],
                         'type': CONST['type']})
const_eval = score_predictions(const_df, diverse_labels_df)

print(f"[FLOOR] T={const_eval['T'].mean():.4f}  S={const_eval['S'].mean():.4f}  "
      f"C={const_eval['C'].mean():.4f}  ->  ACCIDENT score = {accident_score(const_eval):.4f}")

# Tap calibration can bang 4-moi-loai, nen C cua no thap hon con so ma cung hang
# so dat duoc duoi prior tu nhien. Prior cua tap test thuc thi khong biet.
natural_prior = labels_clean['type'].value_counts(normalize=True).max()
print(f"[NOTE] C on this balanced split = {const_eval['C'].mean():.4f}; under the natural "
      f'synthetic prior the same constant would score {natural_prior:.4f}. '
      'The real test prior is unknown.')
```

Lưu ý một chi tiết dễ bỏ sót: cell 83 dùng `diverse_labels_df['rgb_path']` làm cột `path`, chứ
không phải đường dẫn tuyệt đối. `score_predictions` join theo `pathlib.Path(p).stem` nên cả hai
đều hoạt động — nhưng dùng `rgb_path` là bản gốc.

### 8.2 `eval_core` — mẫu ở cell 93, không phải cell 86/87

Đây là điểm dễ nhầm nhất. Notebook 03 cần điểm baseline của **pipeline VLM**, và:

- **Cell 86–87** tạo `eval_baseline` / `eval_final` — nhưng đó là pipeline **classical và
  tracking**, không phải VLM. Dùng sai sẽ so sánh Stage 3 của bạn với một baseline khác hẳn.
- **Cell 93** tạo `eval_vlm` — đúng cái cần, nhưng **chưa từng chạy xong** (OOM ở video 8/20).

Vậy mẫu đúng để dựng `eval_core` là vòng lặp của cell 93, với 2 khác biệt: gọi từng stage thay
vì `run_inference_vlm` (vì bản core của hàm đó bị hỏng), và cache kết quả ra CSV. Bản đã viết sẵn
nằm ở [`HUONG_DAN_03_SPATIAL_S.md`](HUONG_DAN_03_SPATIAL_S.md) Phần 4.2 (Patch A).

### 8.3 Cell 88/89 — chính là cell 4/5 của notebook 03

Cell 88 của file này giống từng dòng với cell 4 của `03_spatial_S.ipynb`, chỉ khác `eval_final`
được đổi tên thành `eval_core`. Cell 89 (đường cong độ nhạy Gaussian) thì giống hệt.

Đây là bằng chứng trực tiếp về nguồn gốc blocker: khi copy 2 cell vẽ biểu đồ sang notebook 03,
người ta bỏ lại 2 cell phía trên (86 và 87) đã tạo ra DataFrame mà chúng cần.

### 8.4 Ba hàm classical cho `run_inference_baseline` của core

Nếu cần `run_inference_baseline` của core chạy được, copy từ file này:

| Hàm | Cell nguồn |
|---|---|
| `predict_accident_time` | 54 |
| `predict_collision_type_clip` + `clip_type_probabilities` | 57 |
| `predict_impact_location_flow` | 61 |

Nhưng cân nhắc: ablation (cell 85) và bảng so sánh (cell 87) đều cho thấy cả 3 tín hiệu này
**thua hằng số của chúng**. `run_inference_baseline` chỉ có giá trị làm mốc so sánh có tài liệu,
không phải thứ để cải tiến.

### 8.5 Section 8 — thứ giá trị nhất để đưa lên `core/`

`core/shared_pipeline.ipynb` kết thúc ở cell 68, không có đường sinh submission. Cell 95–100 của
file này nên được port lên core (qua branch `core/<mô-tả>` riêng theo README Bước 4), vì nó đã có
sẵn 4 thứ mà việc viết lại từ đầu rất dễ làm sai:

1. `SUBSET_N` — dry run trước khi cam kết 20 giờ GPU.
2. Checkpoint mỗi 25 video + resume tự động — điều kiện sống còn với session 12 giờ.
3. Ghép schema với `sample_submission.csv` và điền `CONST` cho dòng không match (thay vì đoán
   giá trị cứng).
4. Check submission suy biến — bắt đúng chế độ lỗi "hằng số đội lốt submission".

---

## Phần 9 — Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `[ERROR] Could not auto-detect the dataset root` | Chưa add competition input hoặc chưa join | Add Input → Competition → `accident`, đồng ý rules trước |
| Bảng audit cell 7 có `exists=False` | Dataset mount layout khác | Đọc log `Contents of /kaggle/input` rồi set `BASE_DIR` tay ở cell 6 |
| `install exit code: 1` | Internet chưa bật | Settings → Internet On |
| `qwen3_vl architecture recognized: False` | transformers cũ vẫn trong bộ nhớ | **Restart Kernel** rồi chạy lại từ cell 0. Upgrade không có hiệu lực với module đã import |
| `Linear4bit layers: 0` | `bitsandbytes` không áp dụng | Model đang fp16 (~16 GB) → sẽ OOM. Kiểm tra `bitsandbytes` đã cài và `DEVICE == 'cuda'` |
| `[CANARY]` trả `''` rồi `AssertionError` | Model load được nhưng không sinh text | **Không đi tiếp.** Thường là NVRTC/JIT hoặc VRAM. Restart Kernel + chạy lại |
| `OutOfMemoryError` ở cell 93 hoặc 96 | Chưa áp Patch 1 | [Phần 6.1](#61-patch-1--sửa-oom-bắt-buộc). Nếu đã áp mà vẫn OOM: hạ `whole_limit` → 16, rồi `image_max_side` → 384 |
| `[WARNING] OOM -- retrying with 24 frames` ở **mọi** video | Frame budget quá lớn cho VRAM còn lại | Stage 1 đang trả lời từ 12 frame trên ~20s (1 frame/1,7s) — có thể bỏ sót va chạm. Điểm đo ra không đáng tin |
| Cả 20 clip cùng một loại va chạm | Collapse của model (đã ghi nhận: `t-bone` 7/7) | Chạy cell 91 xem output thô trước. Nếu parse đúng thì đây là bias prior — xem `02_type_C` |
| Toạ độ ra `(0.00, 0.00)` hoặc `(0.5, 0.5)` cho mọi clip | Stage 3 parse ra rác | Chạy cell 91; kiểm tra thang toạ độ (Qwen3-VL native là [0, 1000]) |
| `AssertionError: Some calibration videos are missing on disk` | Đường dẫn không resolve | Kiểm tra `resolve_video_path` và layout `sim_dataset/videos/<type>/` |
| `AssertionError: Some videos were dropped during inference` (cell 96) | Có clip lỗi decode | Kiểm tra checkpoint xem clip nào thiếu |
| `AssertionError: Submission row count mismatch` (cell 97) | Số dòng không khớp `sample_submission.csv` | Đừng nộp. Kiểm tra `n_unmatched` mà cell in ra |
| Session chết giữa full run | Hết 12 giờ | Bình thường. Chạy lại cell 0→74 rồi cell 96, nó tự resume từ checkpoint |

### Lệnh chẩn đoán VRAM nhanh

```python
# [DEBUG] Trang thai VRAM hien tai
import torch, gc
gc.collect(); torch.cuda.empty_cache()
for g in range(torch.cuda.device_count()):
    free, tot = torch.cuda.mem_get_info(g)
    print(f'GPU{g} {torch.cuda.get_device_name(g)}: '
          f'{(tot-free)/1e9:.2f}/{tot/1e9:.2f} GB dung, {free/1e9:.2f} GB con trong')
print('VLM dang o:', vlm_model.device)
print('Linear4bit layers:', sum(1 for m in vlm_model.modules()
                                if 'Linear4bit' in type(m).__name__))
print('VLM_CFG:', {k: VLM_CFG[k] for k in
                   ('whole_fps', 'whole_limit', 'image_max_side', 'grounding_max_side')})
```

---

## Phụ lục — Bảng tra cứu cell

Các cell bạn sẽ cần tìm lại nhiều nhất:

| Cần gì | Cell | Ghi chú |
|---|---|---|
| Dò `BASE_DIR`, `resolve_video_path` | 6 | Sửa ở đây nếu chạy ngoài Kaggle |
| `pip_install`, cài package | 8 | Comment out `perception_models` để tiết kiệm |
| `labels_df`, `test_df`, `sample_sub`, `real_videos` | 10–12 | `real_videos` là đầu vào của Section 8 |
| `labels_clean` (+ `abs_video_path`) | 21–23 | |
| `sample_frames_stamped`, `vlm_generate`, prompts | 73 | Thêm `vlm_generate_oom_safe` vào đây |
| `SortLiteTracker`, `extract_vehicle_tracks` | 49–50 | |
| `analyze_video_collision`, `FEATURE_COLS` | 52 | |
| `predict_accident_time` | 54 | Core gọi nhưng không định nghĩa |
| `predict_collision_type_clip`, `clip_type_probabilities` | 57 | Core gọi nhưng không định nghĩa |
| `predict_impact_location_flow` | 61 | Core gọi nhưng không định nghĩa |
| `SIGMA_T_LIST`, `SIGMA_X`, `SIGMA_Y`, `CONST` | 63–64 | σ_x = 0.0952, σ_y = 0.1353 |
| **Load Qwen3-VL + `VLM_CFG`** | **72** | **Áp Patch 1 ở đây** |
| Canary | 74 | Chốt an toàn, đừng bỏ qua |
| **3 stage function** | **75** | **Áp Patch 2 ở đây** |
| `run_inference_vlm` | 76 | Bản SẠCH (khác core) |
| `temporal_score`, `spatial_score`, `accident_score`, `score_predictions` | 80 | Hàm điểm chính thức |
| `diverse_videos`, `diverse_labels_df` | 81 | 20 video, seed 42 |
| **`const_eval`** | **83** | Notebook 03 đang thiếu cái này |
| Ablation từng tín hiệu | 85 | Áp Patch 3 để sửa sigma |
| `eval_baseline`, `eval_final` | 86–87 | Pipeline classical/tracking, KHÔNG phải VLM |
| Histogram T/S/C + đường cong độ nhạy | 88–89 | = cell 4/5 của notebook 03 |
| **Dump output thô VLM** | **91** | Cell chẩn đoán quan trọng nhất |
| `eval_vlm` (Qwen3-VL trên calibration) | 93 | OOM ở video 8/20 |
| **Section 8: inference + submission** | **95–100** | Core hoàn toàn không có |
| Conclusion (rất đáng đọc) | 101 | Tổng kết 6 sai sót đã tìm ra và cách sửa |

---

## Tham khảo

- Kaggle: ACCIDENT @ CVPR 2026 — AUTOPILOT Workshop
- Writeup hạng 1: arXiv:2605.29325 — Qwen3-VL, 3 lần gọi mỗi clip, không train, đạt 0.57080.
  Một lần gọi đơn lẻ đã là 0.42238; tách Stage 3 grounding riêng là cải tiến đơn lẻ lớn nhất
  của họ (+0.09356)
- Zhao et al., ICML 2021 — *Calibrate Before Use* (phương pháp đo content-free prior mà
  `02_type_C` áp dụng để chẩn đoán collapse về `t-bone`)
- [`HUONG_DAN_03_SPATIAL_S.md`](HUONG_DAN_03_SPATIAL_S.md) — hướng dẫn cho notebook metric S
- `README.md` — quy ước branch, quy trình PR, quy tắc không sửa core


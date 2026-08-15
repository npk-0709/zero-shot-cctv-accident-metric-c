# Kế hoạch tăng metric S từ 0.3987 lên cao hơn

---

## TL;DR — 7 điều quan trọng nhất

1. **`S = 0.3987` là con số CŨ, đo trên `t_final` sai.** Nó được đo ở cell 94, khi
   `t_final` còn đến từ chuỗi `stage1 (timestamp) → neo classical → stage2 dense`, sai
   **trung bình 3.10 s** so với thời điểm va chạm thật. Sau khi bạn đổi sang NumPro +
   coarse-to-fine frame-số (cell 109/111/113), sai số đó xuống **2.15 s**, và số clip có
   frame grounding nằm trong ±1 s tăng từ **5/20 lên 8/20**. Stage 3 chỉ nhìn **đúng một
   frame tại `t_final`** — nên S gần như chắc chắn đã tự tăng mà bạn chưa đo lại. Chính
   notebook cũng ghi `S=(chua do lai)` ở cell 113.
   → **Việc số 0: đo lại S. Tốn ~3 phút GPU, không sửa một dòng logic nào.**

2. **10% clip trả về toạ độ suy biến `(0.000, 0.000)`** — 2/20 clip calibration, và ít nhất
   1/5 clip trong dry-run trên video thật. Mỗi clip như vậy ăn **S = 0.0000** thay vì ~0.22
   nếu rơi về hằng số. Chỉ cần chặn giá trị suy biến là **+0.02 S**, chi phí GPU = 0.

3. **Dự đoán của Stage 3 bị tán rộng hơn nhãn thật (over-dispersion).** Độ lệch chuẩn của
   toạ độ dự đoán là **(0.235, 0.225)**, còn của nhãn chỉ **(0.130, 0.181)**. Với một hàm
   điểm Gaussian, đây là bằng chứng trực tiếp rằng **co dự đoán về phía prior (shrinkage)
   sẽ tăng điểm** — theo lý thuyết Wiener/James–Stein, và tôi đã kiểm chứng công thức bằng
   mô phỏng. Ước tính **+0.05 → +0.08 S, chi phí GPU = 0** (chỉ hậu xử lý).

4. **Ba việc "miễn phí" ở mục 1–3 cộng lại ước tính đưa S từ 0.399 lên ~0.50–0.52** mà không
   thêm một lần gọi model nào. Đây là phần nên làm trước tiên.

5. **Sau đó, đòn bẩy lớn nhất là coarse-to-fine không gian (crop rồi hỏi lại).** Frame gốc
   1920×1080 bị hạ về 768 px; hộp ground-truth trung bình chỉ **0.095 × 0.135** khung ảnh
   ≈ 73×58 px sau khi hạ, tức khoảng **2×2 patch** của Qwen3-VL. Cắt cửa sổ quanh câu trả
   lời thô rồi hỏi lại chính là cách tăng độ phân giải hiệu dụng gấp 3 lần mà không tăng số
   token. Đây đúng là kỹ thuật vừa hiệu quả cho T (0.4035 → 0.4385).

6. **Cảnh báo về ngân sách.** Dry-run thật đo **49.8 s/video → 28.1 giờ cho 2027 clip**, đã
   vượt 12 h/session và gần hết quota 30 h/tuần của Kaggle. Mỗi lần gọi Stage 3 thêm vào
   cộng khoảng **+1.4 giờ** cho cả lượt chạy. Cấu hình chốt nên **không quá +2 lần gọi**.

7. **Nói thẳng một sự thật ngược ý:** ở điểm hiện tại (T=0.4385, S=0.3987, C=0.25), đạo hàm
   của điểm tổng theo từng thành phần là **C: 0.62, S: 0.24, T: 0.20**. Vì trung bình điều
   hoà, **C đang là mắt yếu nhất và có đòn bẩy gấp 2.5 lần S**. Bạn hỏi về S nên tài liệu
   này tập trung vào S, nhưng nếu mục tiêu là *điểm leaderboard* thì C=0.25 (chỉ hơn mức
   ngẫu nhiên 0.20 một chút, ma trận nhầm lẫn cho thấy mọi thứ dồn về `t-bone`/`sideswipe`)
   là chỗ đáng đầu tư hơn. Chi tiết ở [Phần 8](#phần-8--một-lưu-ý-ngoài-phạm-vi-c-đang-là-mắt-yếu-hơn-s).

---

## Mục lục

- [Phần 1 — Trạng thái hiện tại của notebook](#phần-1--trạng-thái-hiện-tại-của-notebook)
- [Phần 2 — Chẩn đoán: 0.3987 mất điểm ở đâu](#phần-2--chẩn-đoán-03987-mất-điểm-ở-đâu)
- [Phần 3 — Toán học của hàm điểm S (nền tảng cho mọi đề xuất)](#phần-3--toán-học-của-hàm-điểm-s-nền-tảng-cho-mọi-đề-xuất)
- [Phần 4 — Lộ trình 4 tầng, xếp theo lợi ích/chi phí](#phần-4--lộ-trình-4-tầng-xếp-theo-lợi-íchchi-phí)
- [Phần 5 — Code chi tiết từng phương pháp](#phần-5--code-chi-tiết-từng-phương-pháp)
- [Phần 6 — Quy trình đo lường (đọc trước khi tin bất kỳ con số nào)](#phần-6--quy-trình-đo-lường-đọc-trước-khi-tin-bất-kỳ-con-số-nào)
- [Phần 7 — Ngân sách GPU và cấu hình chốt](#phần-7--ngân-sách-gpu-và-cấu-hình-chốt)
- [Phần 8 — Một lưu ý ngoài phạm vi: C đang là mắt yếu hơn S](#phần-8--một-lưu-ý-ngoài-phạm-vi-c-đang-là-mắt-yếu-hơn-s)
- [Phần 9 — Bẫy đã có tiền lệ trong repo, đừng lặp lại](#phần-9--bẫy-đã-có-tiền-lệ-trong-repo-đừng-lặp-lại)
- [Phần 10 — Bảng theo dõi kết quả (điền khi chạy)](#phần-10--bảng-theo-dõi-kết-quả-điền-khi-chạy)

---

## Phần 1 — Trạng thái hiện tại của notebook

### 1.1 Pipeline đang chạy thật (bản chốt ở cell 113)

```
video.mp4
   │
   ├─ stage1_full_scan          (cell 109, bản NumPro)
   │     16 frame @ 2 fps, 448 px, ĐỐT SỐ THỨ TỰ KHUNG (1,2,3...)
   │     → JSON {collision_frame, center_x, center_y, type}
   │     → quy đổi collision_frame về giây qua bảng times[]
   │
   ├─ stage2_time_refine_numbered   (cell 111)
   │     cửa sổ [t−8, t+4] @ 4 fps, 14 frame, đánh số liên tục
   │     → t_final = t_base + 0.35 × clip(t_ref − t_base, ±1.5s)
   │
   ├─ stage3_grounding          (cell 73, KHÔNG đổi từ đầu dự án)   ★ MỤC TIÊU
   │     ĐÚNG 1 frame tại t_final, 768 px, không đốt gì
   │     prompt: {"point": [x, y]} thang [0, 1000]
   │     → GHI ĐÈ hoàn toàn center_x/center_y của Stage 1
   │
   ├─ classify_type_cascade     (cell 76)  — 2–3 lần gọi thêm
   └─ apply_scene_type_postfix  (cell 73)
```

Lưu ý kiến trúc quan trọng: **cell 113 đã bỏ khối "neo classical"**
(`predict_accident_time_ensemble`) khỏi `run_inference_vlm`, vì cell 105 đo được nó
net-negative (T: 0.3519 thô → 0.2782 sau khi neo). Đó là quyết định đúng và tài liệu này
giữ nguyên.

### 1.2 Số đo đã có trong file

| Nguồn | T | S | C | ACCS |
|---|---|---|---|---|
| Constant floor (cell 84) | 0.3800 | 0.2159 | 0.2000 | 0.2446 |
| Tracking (cell 94) | 0.2905 | 0.1497 | 0.4000 | 0.2377 |
| Qwen3-VL **bản cũ** (cell 94) | 0.2724 | **0.3987** | 0.2500 | 0.2947 |
| Stage 1 NumPro đơn lẻ (cell 111) | 0.4035 | — | — | — |
| **+ refine frame-số (cell 111)** | **0.4385** | *chưa đo lại* | *chưa đo lại* | *chưa đo* |

Nếu ghép `T = 0.4385` với `S = 0.3987`, `C = 0.25` thì ACCS = **0.3413**. Nhưng đó là ghép
số từ hai lần chạy khác nhau — **chưa từng có một lần chạy nào đo cả ba cùng lúc trên bản
pipeline chốt**. Đây là lỗ hổng đo lường lớn nhất hiện tại.

### 1.3 Tham số của hàm điểm S

```python
SIGMA_X = mean(x2 - x1) = 0.0952     # chiều rộng bbox trung bình
SIGMA_Y = mean(y2 - y1) = 0.1353     # chiều cao bbox trung bình
S = exp(-0.5 * ((dx/0.0952)^2 + (dy/0.1353)^2))
```

Thống kê nhãn trên 2211 video synthetic:

| | mean | std | min | max |
|---|---|---|---|---|
| `center_x` | 0.4984 | **0.1298** | 0.0552 | 0.9305 |
| `center_y` | 0.4997 | **0.1808** | 0.0176 | 0.9444 |

`CONST = (0.51, 0.51)` (grid-search trên 2211 video).

### 1.4 Môi trường

Kết quả log cell 66: **1 GPU Tesla P100-PCIE-16GB** (17.06 GB, còn 9.47 GB sau khi nạp
model). `whole_limit` đã hạ 32 → 16 ở cell 109. Vẫn thấy `[WARNING] OOM` ở Stage 1 trên
video thật (8/10 clip trong dry-run) — nhưng Stage 3 chỉ 1 frame nên không bị ảnh hưởng.

---

## Phần 2 — Chẩn đoán: 0.3987 mất điểm ở đâu

Đây là toàn bộ toạ độ Stage 3 trả về trên 20 video calibration (từ log cell 94):

```
(0.79,0.44) (0.70,0.28) (0.29,0.55) (0.50,0.60) (0.33,0.56)
(0.60,0.50) (0.00,0.00) (0.25,0.50) (0.51,0.81) (0.41,0.29)
(0.70,0.45) (0.65,0.28) (0.00,0.00) (0.61,0.44) (0.64,0.27)
(0.57,0.75) (0.70,0.60) (0.47,0.72) (0.17,0.63) (0.67,0.67)
```

### 2.1 Nguồn mất điểm A — grounding trên frame SAI (lớn nhất, đã được sửa gián tiếp)

Stage 3 nhìn **đúng một frame tại `t_final`**. Nếu `t_final` lệch 5 giây, model đang được
hỏi "va chạm ở đâu" trên một frame **chưa có va chạm** hoặc **đã tan hiện trường**. Nó vẫn
sẽ trả về một toạ độ — và toạ độ đó vô nghĩa.

So sánh `|t_final − t_gt|` giữa bản cũ (đã dùng để đo S=0.3987) và bản mới ở cell 111:

| | Bản cũ (đo S=0.3987) | Bản mới (NumPro + refine) |
|---|---|---|
| Sai số trung bình | **3.10 s** | **2.15 s** |
| Sai số trung vị | 2.50 s | 1.70 s |
| Clip nằm trong ±0.5 s | 1/20 | **5/20** |
| Clip nằm trong ±1.0 s | 5/20 | **8/20** |
| Clip nằm trong ±2.0 s | 7/20 | **11/20** |

Vài ví dụ cụ thể cho thấy mức độ:

| Video | gt | t_final cũ | t_final mới |
|---|---|---|---|
| `Town03_head-on_night_40` | 6.45 | **15.61** (lệch 9.2 s) | **6.03** (lệch 0.4 s) |
| `Town03_t-bone_rain_23` | 5.60 | **12.77** (lệch 7.2 s) | **5.15** (lệch 0.5 s) |
| `Town05_sideswipe_clear_04` | 5.20 | 10.15 | **5.44** |
| `Town04_sideswipe_wet_10` | 10.05 | 8.05 | **9.97** |

**Kết luận:** S=0.3987 được đo trong điều kiện Stage 3 thường xuyên nhìn sai frame. Con số
thật của Stage 3 hiện tại **cao hơn 0.3987**, chỉ là chưa ai đo. Đây là lý do
[Việc 0](#tầng-0--miễn-phí-hoàn-toàn-làm-trước-mọi-thứ-khác) phải là việc đầu tiên.

### 2.2 Nguồn mất điểm B — toạ độ suy biến `(0.000, 0.000)`

2/20 clip calibration (`Town05_rear-end_rain_142`, `Town10HD_single_sunset_05`) và ít nhất
1/5 clip thật hiển thị trong dry-run (`videos/-AztVDZ6cEE_00.mp4`) trả về đúng góc trên
trái. `spatial_score(0, 0, ~0.5, ~0.5)` ≈ **0.0000**, tức mất trọn 1/20 = 5% của trung bình
mỗi lần.

Ba nguyên nhân có thể, cả ba đều rẻ để chặn:

1. Model trả `{"point": [0, 0]}` khi "không thấy gì" trên frame sai.
2. Regex fallback trong `stage3_grounding` bắt `(0, 0)` từ một câu văn bất kỳ:
   ```python
   m = re.search(r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)', text)
   ```
   Regex này khớp với **mọi** cặp số trong ngoặc, kể cả trong câu giải thích.
3. `_extract_json` dùng `re.search(r'\{.*\}', DOTALL)` — **greedy**. Nếu output có 2 khối
   JSON, nó lấy từ `{` đầu đến `}` cuối và `json.loads` thất bại → trả `None` → dùng lại
   toạ độ Stage 1 (chưa chắc là điều bạn muốn) hoặc rơi vào nhánh regex ở trên.

### 2.3 Nguồn mất điểm C — dự đoán tán rộng hơn nhãn (over-dispersion)

| | std x | std y |
|---|---|---|
| Dự đoán Stage 3 (cả 20) | **0.2346** | **0.2249** |
| Dự đoán Stage 3 (bỏ 2 clip `(0,0)`) | **0.1779** | 0.1673 |
| Nhãn thật (2211 video) | 0.1298 | 0.1808 |

Trục **x bị tán rộng gấp 1.37 lần** nhãn thật ngay cả khi đã bỏ hai clip suy biến. Trục y
thì không (0.167 < 0.181).

Ý nghĩa: nếu `pred = gt + noise` với `noise` độc lập, thì
`Var(pred) = Var(gt) + Var(noise)`. Tán rộng hơn nhãn ⇒ có nhiễu thật ⇒ và với một hàm
điểm Gaussian, **co dự đoán về phía prior làm giảm sai số bình phương, tức tăng điểm kỳ
vọng**. Chi tiết toán ở [Phần 3.3](#33-shrinkage-về-prior-tại-sao-nó-tăng-điểm-và-đặt-λ-bằng-bao-nhiêu).

### 2.4 Nguồn mất điểm D — độ phân giải hiệu dụng quá thấp

`grounding_max_side = 768`. Với clip synthetic 1920×1080 → resize về 768×432.
Hộp ground-truth trung bình `0.0952 × 0.1353` khung ảnh → **73 × 58 px** trên ảnh đã hạ.
Qwen-VL gộp mỗi 28×28 px thành một visual token ⇒ mục tiêu chỉ chiếm khoảng **2.6 × 2.1
patch**.

Với clip thật thì tệ hơn: `test_metadata.csv` có clip **3840×2160** (ví dụ
`UarP8qU1S-c_00.mp4`), hạ về 768 là tỉ lệ 1/5 → mục tiêu còn khoảng 1×1 patch.

Đây chính là lý do coarse-to-fine không gian (crop rồi hỏi lại) là hướng có tiềm năng lớn
nhất trong nhóm "phải trả thêm GPU".

### 2.5 Bảng tổng hợp chẩn đoán

| # | Nguồn mất điểm | Bằng chứng | Chi phí sửa | Ước tính lợi ích |
|---|---|---|---|---|
| A | Grounding trên frame sai | `|Δt|` 3.10 s → 2.15 s | 0 (đã sửa, chỉ cần đo lại) | **lớn, chưa biết** |
| B | Toạ độ suy biến `(0,0)` | 2/20 + ≥1/5 clip thật | 0 GPU | **+0.02** |
| C | Over-dispersion | std x 0.235 vs 0.130 | 0 GPU | **+0.05 → +0.08** |
| D | Độ phân giải hiệu dụng | GT box ≈ 2×2 patch | +1 lần gọi | **+0.03 → +0.10** (cần đo) |
| E | Chỉ 1 frame, không bỏ phiếu | — | +2 lần gọi | **+0.03 → +0.08** (cần đo) |

---

## Phần 3 — Toán học của hàm điểm S (nền tảng cho mọi đề xuất)

Phần này không phải lý thuyết trang trí — ba con số ở đây quyết định thứ tự ưu tiên trong
Phần 4.

### 3.1 Độ nhạy: sai bao nhiêu thì mất bao nhiêu điểm

`S = exp(-0.5 * ((dx/0.0952)² + (dy/0.1353)²))`

| Sai lệch | Chỉ theo x | Chỉ theo y | Cả hai trục |
|---|---|---|---|
| 0.02 (≈38 px/1920) | 0.978 | 0.989 | 0.967 |
| 0.05 (≈96 px) | 0.871 | 0.933 | 0.813 |
| 0.10 (≈192 px) | 0.576 | 0.758 | 0.437 |
| 0.15 | 0.291 | 0.541 | 0.157 |
| 0.20 | 0.107 | 0.328 | 0.035 |

**Đọc bảng này một lần rồi ghi nhớ:** phải đúng trong khoảng **10% chiều rộng khung** mới
có điểm khá. Sai 20% thì gần như bằng 0. Và vì `σ_y > σ_x` gấp 1.42 lần, **sai theo trục
ngang đắt hơn sai theo trục dọc** — nếu phải chọn, hãy tối ưu độ chính xác theo x.

### 3.2 Điểm kỳ vọng khi sai số là Gaussian

Nếu sai số dự đoán là Gaussian với phương sai `v_x, v_y` thì tích phân đóng:

```
E[S] = 1 / sqrt( (1 + v_x/σ_x²) · (1 + v_y/σ_y²) )
```

Công thức này biến mọi câu hỏi "cải thiện độ chính xác bao nhiêu thì được bao nhiêu điểm"
thành một phép tính:

| Sai số (std đẳng hướng) | E[S] |
|---|---|
| 0 (hoàn hảo) | 1.0000 |
| 0.03 | 0.9311 |
| 0.05 | 0.8304 |
| 0.07 | 0.7156 |
| 0.10 | 0.5545 |
| 0.13 | 0.4260 |
| 0.16 | 0.3302 |
| 0.20 | 0.2408 |
| = độ tán của nhãn (constant) | **0.3543** |

Lưu ý: **hằng số cho E[S] = 0.3543 trên toàn bộ 2211 video**, nhưng chỉ đạt **0.2159** trên
20 video calibration. Nghĩa là **tập 20 video này khó hơn mức trung bình đối với S**. Mọi
so sánh phải làm trên cùng tập này, và đừng ngoại suy trực tiếp sang leaderboard.

### 3.3 Shrinkage về prior: tại sao nó tăng điểm, và đặt λ bằng bao nhiêu

Đây là đề xuất "miễn phí" quan trọng nhất, nên tôi trình bày đầy đủ.

**Thiết lập.** Gọi `μ` là prior (`CONST = 0.51`), `p` là dự đoán của model, và dùng thay
vào đó:

```
q = μ + λ · (p − μ)          với λ ∈ (0, 1]
```

Nếu `gt ~ N(μ, τ²)` và `p = gt + ε` với `ε ~ N(0, s²)` thì sai số của `q` là

```
q − gt = (λ−1)(gt−μ) + λ·ε
Var = (1−λ)²τ² + λ²s²
```

Cực tiểu tại **λ\* = τ² / (τ² + s²)**, và phương sai còn lại `v_min = λ\* · s²`. Vì E[S] ở
mục 3.2 **giảm đơn điệu theo v**, cực tiểu phương sai chính là cực đại điểm kỳ vọng. Đây
là công thức Wiener / co ngót James–Stein, không phải trick.

**Lợi ích theo mức nhiễu:**

| s/τ (nhiễu so với độ tán nhãn) | λ\* | E[S] thô | E[S] sau shrink | Lợi |
|---|---|---|---|---|
| 0.4 | 0.862 | 0.7743 | 0.7991 | +0.025 |
| 0.6 | 0.735 | 0.6039 | 0.6746 | +0.071 |
| 0.8 | 0.610 | 0.4616 | 0.5844 | **+0.123** |
| 1.0 | 0.500 | 0.3543 | 0.5233 | **+0.169** |

**Model càng kém, shrinkage càng có giá.** Ở mức S≈0.40 hiện tại, s/τ vào khoảng 0.6–0.8,
tức vùng lợi +0.07 → +0.12.

**Cách đặt λ mà KHÔNG cần ground-truth từng video** (điểm đắt giá nhất của mục này):

```
Var(p) = Var(gt) + s²   ⇒   λ* = Var(gt) / Var(p)
```

`Var(gt)` lấy từ `labels_clean` (đã biết: 0.1298², 0.1808²), `Var(p)` tính từ chính các dự
đoán. Tôi đã kiểm chứng estimator này bằng mô phỏng — khớp đến 3 chữ số:

```
s/tau=0.4  lam_true=0.862  lam_hat=0.864
s/tau=0.6  lam_true=0.735  lam_hat=0.735
s/tau=0.8  lam_true=0.610  lam_hat=0.610
s/tau=1.0  lam_true=0.500  lam_hat=0.500
```

Hệ quả rất mạnh: **bạn có thể ước lượng λ trực tiếp trên 2027 dự đoán của tập TEST, không
cần nhãn.** Điều này tránh hoàn toàn nguy cơ overfit λ vào 20 video CARLA, và tự thích ứng
nếu phân bố toạ độ trên CCTV thật khác CARLA.

Áp vào số đo hiện có:

| Tập | Var(p) | λ theo moment |
|---|---|---|
| Cả 20 clip | (0.0550, 0.0506) | (0.31, 0.65) |
| Bỏ 2 clip `(0,0)` | (0.0317, 0.0280) | (0.53, 1.00) |

Hai cột này khác nhau nhiều — đúng như dự đoán, vì hai điểm `(0,0)` là outlier chứ không
phải nhiễu Gaussian. **Nên chặn suy biến TRƯỚC, rồi mới ước lượng λ.** Sau khi chặn, λ_x
≈ 0.53 và λ_y ≈ 1.0 (không cần co trục y).

### 3.4 Ước tính tổng hợp (mô phỏng Monte-Carlo hiệu chỉnh theo số đo thật)

Tôi hiệu chỉnh một thế giới mô phỏng sao cho nó tái tạo đúng hai con số đã đo trên chính
20 video này (hằng số = 0.2159, grounding = 0.3987), rồi thử từng can thiệp:

| Can thiệp | S ước tính |
|---|---|
| Hiện tại | 0.3987 |
| + chặn suy biến `(0,0)` → rơi về `CONST` | **0.4202** |
| + shrinkage λ=0.7–0.8 (giữ nguyên suy biến) | 0.4623 – 0.4661 |
| **+ chặn suy biến VÀ shrinkage λ=0.7–0.8** | **0.5129 – 0.5148** |
| + thêm bỏ phiếu 3 khung (ρ=0.5) + shrinkage | 0.5947 |
| + bỏ phiếu 3 khung (ρ=0.3, lạc quan) + shrinkage | 0.6384 |

> **Đây là ước tính, không phải số đo.** Mô phỏng giả định nhiễu Gaussian; thực tế có đuôi
> dày. Cột "bỏ phiếu 3 khung" phụ thuộc mạnh vào ρ — độ tương quan giữa các lần gọi. Vì
> greedy decode và 3 frame liền kề rất giống nhau, ρ thực tế có thể là 0.6–0.8, không phải
> 0.3. Toàn bộ Phần 6 tồn tại để bạn thay ước tính này bằng số đo thật.
>
> Điều **không** phụ thuộc mô phỏng: hướng của hiệu ứng (shrinkage tăng điểm khi
> over-dispersion), và giá trị λ\* (có công thức đóng, đã kiểm chứng).

---

## Phần 4 — Lộ trình 4 tầng, xếp theo lợi ích/chi phí

### Tầng 0 — miễn phí hoàn toàn, làm trước mọi thứ khác

| # | Việc | Chi phí GPU | Ước tính |
|---|---|---|---|
| **0.1** | **Đo lại S trên `t_final` mới** (NumPro + refine frame-số) | ~3 phút (20 lần gọi Stage 3) | reset baseline |
| **0.2** | Dựng cache `t_final`/`type` + harness đánh giá S | 1 lần ~15 phút | tiết kiệm mọi lần sau |
| **0.3** | Chẩn đoán: dump output thô, đếm `None`/`(0,0)`, đo bias + độ tán, so `S_oracle` vs `S_pipeline` | ~3 phút | quyết định hướng |

**Không được bỏ qua 0.1.** Mọi con số so sánh phía sau đều phải so với baseline mới này,
không phải với 0.3987.

### Tầng 1 — hậu xử lý, 0 lần gọi thêm

| # | Việc | Ước tính | Rủi ro |
|---|---|---|---|
| **1.1** | Parser cứng hơn: bỏ regex bắt bừa, `_extract_json` không greedy, hỗ trợ nhiều schema | +0.00 → +0.03 | thấp |
| **1.2** | Chặn suy biến `(0,0)` + kẹp vào dải phân vị nhãn, fallback về `CONST` | **+0.02** | thấp |
| **1.3** | Shrinkage λ (đặt theo moment, xác nhận bằng sweep) | **+0.05 → +0.08** | trung bình (xem 9.2) |
| **1.4** | Hiệu chỉnh lệch hệ thống (trừ `mean(pred − gt)`) | +0.00 → +0.03 | trung bình (2 tham số / n=20) |
| **1.5** | Hợp nhất robust toạ độ Stage 1 + Stage 3 (median/trọng số) | +0.00 → +0.03 | thấp |

### Tầng 2 — vẫn 1 lần gọi (chỉ đổi prompt/ảnh, chi phí không tăng)

| # | Việc | Ước tính | Rủi ro |
|---|---|---|---|
| **2.1** | Hỏi `bbox_2d` rồi lấy tâm, thay vì hỏi `point` | +0.00 → +0.05 | thấp |
| **2.2** | Prompt có điều kiện loại va chạm (đảo cascade lên trước grounding) | +0.01 → +0.04 | thấp |
| **2.3** | Đưa 3 frame ngữ cảnh vào **cùng một** lần gọi, hỏi điểm ở frame giữa | +0.00 → +0.04 | thấp |
| **2.4** | Sweep `grounding_max_side` ∈ {768, 896, 1024} | +0.00 → +0.02 | rất thấp |
| **2.5** | Crop quanh **toạ độ Stage 1** rồi hỏi (zoom mà không thêm lần gọi) | +0.02 → +0.06 | trung bình (map toạ độ) |

### Tầng 3 — trả thêm GPU

| # | Việc | Chi phí | Ước tính | Rủi ro |
|---|---|---|---|---|
| **3.1** | Coarse-to-fine: hỏi thô → crop → hỏi lại → map về khung gốc | +1 gọi (+1.4 h) | **+0.03 → +0.10** | trung bình |
| **3.2** | Bỏ phiếu 3 khung (t−0.25, t, t+0.25) + median + shrinkage thích ứng theo độ tán | +2 gọi (+2.8 h) | +0.03 → +0.08 | thấp |
| **3.3** | Set-of-Mark: YOLO đánh số xe, hỏi "hai xe nào va vào nhau" → điểm tiếp xúc | +1 gọi (+1.4 h) | −0.05 → +0.10 | **cao** (xem 9.3) |
| **3.4** | Flip TTA (lật ngang, hỏi, lật lại, lấy trung bình) | +1 gọi (+1.4 h) | +0.00 → +0.03 | thấp |
| **3.5** | YOLO snap có ngưỡng (kéo điểm về xe gần nhất nếu đủ gần) | ~0 GPU | +0.00 → +0.01 | thấp |

### Thứ tự thực hiện tôi khuyến nghị

```
1. Tầng 0 toàn bộ                        (~25 phút GPU)  → có baseline thật + biết mất điểm ở đâu
2. 1.1 + 1.2                             (0 GPU)         → chốt ngay, gần như không rủi ro
3. 1.3 (shrinkage)                       (0 GPU)         → phần lợi lớn nhất/rẻ nhất
4. 2.1, 2.2, 2.3, 2.4 — A/B từng cái     (~3 phút/cái)   → chọn biến thể prompt tốt nhất
5. 3.1 (coarse-to-fine)                  (~6 phút)       → nếu +≥0.03 thì giữ
6. 3.2 (bỏ phiếu) — CHỈ nếu còn ngân sách GPU
7. Đo lại T/S/C CÙNG LÚC trên bản chốt, rồi mới chạy full
```

**Điểm dừng:** khi tổng ngân sách vượt 2 lần gọi Stage 3 thêm (28.1 h → 30.9 h), hãy dừng
và chốt. Quota Kaggle là 30 h/tuần.

---

## Phần 5 — Code chi tiết từng phương pháp

Toàn bộ code dưới đây **chỉ THÊM cell mới vào cuối notebook**, không sửa/xoá cell nào —
đúng theo cách bạn đã làm ở cell 109/111/113. Python dùng định nghĩa mới nhất, nên các
hàm này sẽ ghi đè bản cũ một cách an toàn.

### 5.0 — Cell A: cache `t_final` + `type`, chạy MỘT LẦN

```python
# [CACHE] Chay MOT LAN: Stage 1 (NumPro) + Stage 2 (frame-so) cho 20 video
# calibration, luu ra CSV. Moi thi nghiem Stage 3 sau day chi doc lai file nay
# -- Stage 1/2 dung greedy decode (do_sample=False) nen ket qua TAT DINH, chay
# lai chung chi ton GPU chu khong doi gi. Vong lap goc ton ~15 phut moi lan thu
# mot y tuong Stage 3; voi cache chi con ~3 phut.
import pathlib, time

CACHE_S12 = OUTPUT_DIR / 'rows_stage12_numpro.csv'


def _duration_of(vp):
    cap = cv2.VideoCapture(str(vp))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (n / fps) if fps > 0 else 20.0


if CACHE_S12.exists():
    rows_s12 = pd.read_csv(CACHE_S12).to_dict('records')
    print(f'[CACHE] doc lai {len(rows_s12)} dong tu {CACHE_S12}')
else:
    rows_s12, _t0 = [], time.time()
    for _i, vp in enumerate(diverse_videos, 1):
        duration = _duration_of(vp)
        scene = SCENE_BY_PATH.get('videos/' + vp.name)      # None tren synthetic
        pred = stage1_full_scan(vp, duration, scene)        # ban NumPro (cell 109)
        t_final = stage2_time_refine_numbered(vp, pred['accident_time'], duration)
        rows_s12.append({
            'path': str(vp), 'duration': duration, 't_final': t_final,
            't_stage1': pred['accident_time'],
            'x_stage1': pred['center_x'], 'y_stage1': pred['center_y'],
            'type_stage1': pred['type'],
        })
        print(f"[{_i:2d}/{len(diverse_videos)}] {vp.name:42s} "
              f"t_final={t_final:6.2f}  xy1=({pred['center_x']:.2f},{pred['center_y']:.2f})  "
              f"({time.time()-_t0:5.0f}s)")
    pd.DataFrame(rows_s12).to_csv(CACHE_S12, index=False)
    print(f'[CACHE] da luu {CACHE_S12}')

# Gan ground truth vao tung dong -- can t_gt de do S_ORACLE (grounding tai thoi
# diem THAT), tach bach chat luong Stage 3 khoi sai so cua Stage 1/2.
_gt = diverse_labels_df.copy()
_gt['stem'] = _gt['rgb_path'].map(lambda p: pathlib.Path(p).stem)
_gt = _gt.set_index('stem')
for r in rows_s12:
    _st = pathlib.Path(r['path']).stem
    r['t_gt'] = float(_gt.loc[_st, 'accident_time'])
    r['x_gt'] = float(_gt.loc[_st, 'center_x'])
    r['y_gt'] = float(_gt.loc[_st, 'center_y'])
    r['type_gt'] = str(_gt.loc[_st, 'type'])

print(f"[STATUS] cache san sang: {len(rows_s12)} video | "
      f"|t_final - t_gt| trung binh = "
      f"{np.mean([abs(r['t_final'] - r['t_gt']) for r in rows_s12]):.2f}s")
```

### 5.1 — Cell B: harness đánh giá S (dùng cho mọi thí nghiệm)

```python
# [EVAL] Harness do S. Moi bien the grounding co chu ky (video_path, t, row) va
# tra ve None / (x, y) / (x, y, spread). 'spread' la do tan giua nhieu lan goi,
# dung cho shrinkage thich ung o muc 3.2.
#
# use_gt_time=True do S_ORACLE: grounding tai thoi diem va cham THAT. Con so do
# la TRAN cua Stage 3 hien tai. Khoang cach giua S_ORACLE va S_PIPELINE chinh la
# phan diem con phu thuoc vao viec cai thien T -- rat dang biet truoc khi bo
# cong toi uu prompt.

def eval_S(grounding_fn, name='', use_gt_time=False, rows=None,
           lam=(1.0, 1.0), bias=(0.0, 0.0), fallback='const',
           lam_lo=None, spread_gate=None, verbose=True):
    rows = rows_s12 if rows is None else rows
    mu_x, mu_y = CONST['center_x'], CONST['center_y']
    out = []
    for r in rows:
        vp = pathlib.Path(r['path'])
        t = float(r['t_gt'] if use_gt_time else r['t_final'])

        res = grounding_fn(vp, t, r)
        spread, used = np.nan, 'model'
        if isinstance(res, tuple) and len(res) == 3:
            res, spread = (res[0], res[1]), res[2]

        if res is None:
            used = fallback
            res = ((mu_x, mu_y) if fallback == 'const'
                   else (float(r['x_stage1']), float(r['y_stage1'])))

        # shrinkage thich ung: neu nhieu lan goi khong dong y (spread lon) thi co
        # manh hon, vi do tan chinh la mot uoc luong truc tiep cua nhieu
        lx, ly = lam
        if lam_lo is not None and spread_gate is not None and np.isfinite(spread) \
                and spread > spread_gate:
            lx, ly = lam_lo

        x = _clip01(mu_x + lx * (res[0] + bias[0] - mu_x))
        y = _clip01(mu_y + ly * (res[1] + bias[1] - mu_y))

        out.append({'stem': vp.stem, 'type_gt': r['type_gt'],
                    'x_raw': res[0], 'y_raw': res[1], 'spread': spread,
                    'x': x, 'y': y, 'used': used,
                    'S': spatial_score(x, y, r['x_gt'], r['y_gt']),
                    'dx': x - r['x_gt'], 'dy': y - r['y_gt']})
    df = pd.DataFrame(out)

    # Bootstrap CI: n=20 thi sai so chuan rat lon, mot con so tran khong noi len
    # dieu gi. Luon doc khoang, khong doc chu so.
    _rng = np.random.default_rng(0)
    boot = [df['S'].sample(len(df), replace=True, random_state=int(_rng.integers(1e9))).mean()
            for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    if verbose:
        print(f"[S] {name or grounding_fn.__name__:34s} "
              f"{'ORACLE' if use_gt_time else 'PIPELINE'}  "
              f"S = {df['S'].mean():.4f}  (95% CI {lo:.4f}-{hi:.4f})  "
              f"fallback {int((df['used'] != 'model').sum())}/{len(df)}")
    return df


def compare_S(df_new, df_base, name='moi'):
    """In bang delta tung video -- de biet cai thien la HE THONG hay chi may man
    o 1-2 video. Voi n=20, mot video tu 0.02 len 0.95 da du day trung binh len
    +0.046 ma khong chung minh duoc gi."""
    m = df_base[['stem', 'type_gt', 'S']].rename(columns={'S': 'S_base'}).merge(
        df_new[['stem', 'S']].rename(columns={'S': 'S_new'}), on='stem')
    m['delta'] = m['S_new'] - m['S_base']
    print(f"\n[{name}] S {df_base['S'].mean():.4f} -> {df_new['S'].mean():.4f}  "
          f"({df_new['S'].mean() - df_base['S'].mean():+.4f})  |  "
          f"tot len {int((m['delta'] > 0.01).sum())}/{len(m)} video, "
          f"te di {int((m['delta'] < -0.01).sum())}/{len(m)}")
    display(m.sort_values('delta').round(4))
    return m


# Bao lai ban goc cho khop chu ky (video_path, t, row)
def g_base(vp, t, row=None):
    return stage3_grounding(vp, t)


eval_base_pipe = eval_S(g_base, 'stage3_grounding (goc)')
eval_base_orac = eval_S(g_base, 'stage3_grounding (goc)', use_gt_time=True)

# Constant floor tren dung 20 video nay, de co moc so sanh
_const_S = np.mean([spatial_score(CONST['center_x'], CONST['center_y'],
                                 r['x_gt'], r['y_gt']) for r in rows_s12])
print(f'\n[FLOOR]  constant (0.51, 0.51)  S = {_const_S:.4f}')
print(f'[CU]     bao cao o cell 94        S = 0.3987  (do voi t_final CU)')
print(f'[TRAN]   S_ORACLE - S_PIPELINE    = '
      f"{eval_base_orac['S'].mean() - eval_base_pipe['S'].mean():+.4f}  "
      '<- phan diem con phu thuoc vao chat luong T')
```

### 5.2 — Cell C: chẩn đoán trước khi sửa

```python
# [DIAG] Dump output THO + do bias/do tan. Lam viec nay TRUOC khi toi uu prompt:
# rat nhieu 'cai tien' thuc chat chi la sua loi parse.
print('=' * 78)
for r in rows_s12[:6]:
    vp, t = pathlib.Path(r['path']), float(r['t_final'])
    frames = sample_frames_stamped(vp, 1.0, t, t + 1e-3, limit=1,
                                   max_side=VLM_CFG['grounding_max_side'], burn=False)
    raw = vlm_generate_oom_safe(GROUNDING_PROMPT, frames, 64, min_frames=1)
    print(f"{vp.stem:40s} t={t:6.2f} (gt {r['t_gt']:5.2f})")
    print(f"   RAW    : {raw!r}")
    print(f"   parsed : {_extract_json(raw)}")
    print(f"   gt xy  : ({r['x_gt']:.3f}, {r['y_gt']:.3f})")
print('=' * 78)

# 1. Bao nhieu lan suy bien / parse fail
_d = eval_base_pipe
_n_zero = int(((_d['x_raw'].abs() < 0.02) & (_d['y_raw'].abs() < 0.02)).sum())
print(f"\n[DIAG] toa do (0,0)      : {_n_zero}/{len(_d)}")
print(f"[DIAG] fallback (None)   : {int((_d['used'] != 'model').sum())}/{len(_d)}")
print(f"[DIAG] gia tri x phan biet: {_d['x_raw'].round(3).nunique()}/{len(_d)}  "
      f"| y: {_d['y_raw'].round(3).nunique()}/{len(_d)}")

# 2. Lech he thong (bias) -- neu mean(dx) khac 0 dang ke thi tru di la mien phi
print(f"\n[DIAG] bias  mean(dx)={_d['dx'].mean():+.4f}  mean(dy)={_d['dy'].mean():+.4f}")
print(f"[DIAG] sai so std(dx)={_d['dx'].std():.4f}  std(dy)={_d['dy'].std():.4f}"
      f"   (so sanh: sigma_x={SIGMA_X:.4f}, sigma_y={SIGMA_Y:.4f})")

# 3. Over-dispersion -> lambda theo phuong phap moment (khong can gt tung video)
_tau_x = float(labels_clean['center_x'].std())
_tau_y = float(labels_clean['center_y'].std())
_ok = _d[(_d['x_raw'].abs() >= 0.02) | (_d['y_raw'].abs() >= 0.02)]   # bo suy bien
_vx, _vy = _ok['x_raw'].var(ddof=1), _ok['y_raw'].var(ddof=1)
print(f"\n[DIAG] std du doan  x={np.sqrt(_vx):.4f}  y={np.sqrt(_vy):.4f}")
print(f"[DIAG] std nhan     x={_tau_x:.4f}  y={_tau_y:.4f}")
print(f"[DIAG] lambda* theo moment = ({min(1.0, _tau_x**2/_vx):.3f}, "
      f"{min(1.0, _tau_y**2/_vy):.3f})   <- dung lam diem khoi dau cho muc 1.3")
```

Cách đọc kết quả:

| Quan sát | Chẩn đoán | Đi tới |
|---|---|---|
| `S_ORACLE − S_PIPELINE` > 0.08 | Điểm S còn bị T kéo xuống nhiều | Ưu tiên tiếp T, và mục 3.2 (bỏ phiếu nhiều khung bù cho `t_final` lệch) |
| Nhiều `(0,0)` hoặc `None` | Lỗi parse / model bỏ cuộc | 1.1 + 1.2 |
| `std(pred) > std(nhãn)` | Over-dispersion | 1.3 |
| `mean(dx)` hoặc `mean(dy)` lệch > 0.03 | Lệch hệ thống | 1.4 |
| Ít giá trị phân biệt (≤ 5/20) | Model trả prior, không đọc ảnh | 2.1, 2.2, 3.3 |
| Sai số lớn nhưng phân tán đều | Model đọc ảnh, chỉ chưa đủ nét | 2.4, 2.5, 3.1 |

### 5.3 — Mục 1.1 + 1.2: parser cứng hơn + chặn suy biến

```python
# [FIX] Parser toa do cung hon + chan gia tri suy bien.
#
# Ba loi cu the cua ban goc:
#  1. _extract_json dung re.search(r'\{.*\}', DOTALL) -- GREEDY. Output co 2 khoi
#     JSON se lay tu '{' dau den '}' cuoi va json.loads that bai -> tra None.
#  2. Regex fallback r'\((\d+...),\s*(\d+...)\)' khop MOI cap so trong ngoac, ke
#     ca trong cau giai thich -- day la mot duong sinh ra (0, 0).
#  3. Quy doi thang [0,1000] bang 'if x > 1.0 or y > 1.0' vo hieu khi model tra
#     ve [0, 0] tren thang 1000 (khong co gia tri nao > 1) hoac tra ve dang tron
#     lan [700, 0.4].

def _extract_json_all(text):
    """Tra ve MOI dict JSON tim duoc, tu khoi ngan nhat den dai nhat."""
    out = []
    for m in re.finditer(r'\{', text or ''):
        for end in re.finditer(r'\}', text[m.start():]):
            frag = text[m.start(): m.start() + end.end()]
            try:
                obj = json.loads(frag)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
                break
    return out


# Dai hop le cua toa do, lay tu phan vi nhan thay vi doan. Nhan that co
# center_y thap nhat 0.0176 nen KHONG chan cung o 0.05 -- dung phan vi.
XY_LO_X, XY_HI_X = (float(labels_clean['center_x'].quantile(0.005)),
                    float(labels_clean['center_x'].quantile(0.995)))
XY_LO_Y, XY_HI_Y = (float(labels_clean['center_y'].quantile(0.005)),
                    float(labels_clean['center_y'].quantile(0.995)))
print(f'[STATUS] dai toa do hop le: x[{XY_LO_X:.3f},{XY_HI_X:.3f}] '
      f'y[{XY_LO_Y:.3f},{XY_HI_Y:.3f}]')


def _to_unit(x, y):
    """Quy doi ve [0,1] dua tren DO LON CUA CA CAP, khong xet tung so rieng le."""
    if max(abs(x), abs(y)) > 1.5:            # ro rang la thang [0,1000] (hoac [0,100])
        scale = 1000.0 if max(abs(x), abs(y)) > 100.0 else 100.0
        return x / scale, y / scale
    return x, y


def parse_point(text):
    """Bat toa do tu output tho. Tra None khi khong chac -- 'khong biet' tot hon
    'doan (0,0)', vi caller se rot ve prior thay vi ve goc tren trai."""
    for j in _extract_json_all(text):
        cand = None
        for key in ('point', 'point_2d', 'center', 'impact_point'):
            v = j.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                cand = (_safe_float(v[0], np.nan), _safe_float(v[1], np.nan))
                break
        if cand is None and 'center_x' in j and 'center_y' in j:
            cand = (_safe_float(j['center_x'], np.nan), _safe_float(j['center_y'], np.nan))
        if cand is None:
            v = j.get('points')
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
                cand = (_safe_float(v[0][0], np.nan), _safe_float(v[0][1], np.nan))
        if cand and all(np.isfinite(c) for c in cand):
            return _to_unit(*cand)

    # Chi con nhan dang '[x, y]' hoac '(x, y)' khi KHONG co JSON nao -- va bat
    # buoc phai la 2 so tach nhau boi dau phay, nam sat nhau.
    m = re.search(r'[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]', text or '')
    if m:
        return _to_unit(float(m.group(1)), float(m.group(2)))
    return None


def sanitize_point(pt):
    """None neu suy bien; nguoc lai kep vao dai phan vi nhan."""
    if pt is None:
        return None
    x, y = pt
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    # (0,0) va (1,1): gia tri model tra khi 'khong thay gi', da quan sat 2/20 clip
    # calibration va >=1/5 clip that. Khong phai du doan -- la co che that bai.
    if (abs(x) < 0.02 and abs(y) < 0.02) or (x > 0.98 and y > 0.98):
        return None
    return (float(np.clip(x, XY_LO_X, XY_HI_X)), float(np.clip(y, XY_LO_Y, XY_HI_Y)))


def g_v1(vp, t, row=None):
    """stage3_grounding voi parser cung hon + chan suy bien. Cung 1 lan goi."""
    frames = sample_frames_stamped(vp, 1.0, t, t + 1e-3, limit=1,
                                   max_side=VLM_CFG['grounding_max_side'], burn=False)
    if not frames:
        return None
    return sanitize_point(parse_point(vlm_generate_oom_safe(GROUNDING_PROMPT, frames,
                                                           64, min_frames=1)))


eval_v1 = eval_S(g_v1, 'v1: parser + chan suy bien')
compare_S(eval_v1, eval_base_pipe, 'v1')
```

### 5.4 — Mục 1.3: shrinkage (không cần gọi lại model)

```python
# [POST] Shrinkage ve prior. KHONG can GPU: tinh lai tu toa do da cache trong
# eval_v1. Ly thuyet o Phan 3.3 -- lam* = tau^2/(tau^2 + s^2), va vi
# E[S] = 1/sqrt(prod(1 + v/sigma^2)) giam don dieu theo v, cuc tieu phuong sai
# chinh la cuc dai diem ky vong.

def rescore(df, lam, bias=(0.0, 0.0)):
    mu_x, mu_y = CONST['center_x'], CONST['center_y']
    xs = np.clip(mu_x + lam[0] * (df['x'] + bias[0] - mu_x), 0, 1)
    ys = np.clip(mu_y + lam[1] * (df['y'] + bias[1] - mu_y), 0, 1)
    gt = pd.DataFrame(rows_s12)
    gt['stem'] = gt['path'].map(lambda p: pathlib.Path(p).stem)
    gt = gt.set_index('stem').loc[df['stem']]
    return float(np.mean([spatial_score(a, b, gx, gy) for a, b, gx, gy
                          in zip(xs, ys, gt['x_gt'], gt['y_gt'])]))


print('[SWEEP] lambda dong nhat 2 truc')
for _lam in (1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5):
    print(f'   lam={_lam:.2f}  S={rescore(eval_v1, (_lam, _lam)):.4f}')

print('\n[SWEEP] lambda rieng tung truc (sigma_y > sigma_x nen 2 truc khong doi xung)')
_best = (None, -1)
for _lx in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
    _row = []
    for _ly in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        _s = rescore(eval_v1, (_lx, _ly))
        _row.append(f'{_s:.4f}')
        if _s > _best[1]:
            _best = ((_lx, _ly), _s)
    print(f'   lam_x={_lx:.1f} | ' + '  '.join(_row))
print(f'\n[BEST sweep] lam={_best[0]}  S={_best[1]:.4f}')

# Uoc luong doc lap KHONG dung ground truth -- doi chieu voi sweep. Neu hai con
# so lech xa, hay tin ban moment: sweep tren n=20 rat de overfit.
_ok = eval_v1[eval_v1['used'] == 'model']
_lam_mom = (min(1.0, float(labels_clean['center_x'].std())**2 / _ok['x_raw'].var(ddof=1)),
            min(1.0, float(labels_clean['center_y'].std())**2 / _ok['y_raw'].var(ddof=1)))
print(f'[MOMENT   ] lam={tuple(round(v, 3) for v in _lam_mom)}  '
      f'S={rescore(eval_v1, _lam_mom):.4f}')

# Bias: tru lech he thong. CHI ap dung neu |mean| > 1 sai so chuan cua trung binh,
# nguoc lai la dang fit nhieu tren n=20.
_bx, _by = eval_v1['dx'].mean(), eval_v1['dy'].mean()
_sx, _sy = eval_v1['dx'].sem(), eval_v1['dy'].sem()
_bias = (-_bx if abs(_bx) > _sx else 0.0, -_by if abs(_by) > _sy else 0.0)
print(f'[BIAS     ] mean(dx)={_bx:+.4f}+-{_sx:.4f}  mean(dy)={_by:+.4f}+-{_sy:.4f}'
      f'  -> ap dung {tuple(round(v, 4) for v in _bias)}')
print(f'[BIAS     ] S={rescore(eval_v1, _lam_mom, _bias):.4f}')
```

Sau khi chốt được `lam` và `bias`, đưa thẳng vào `eval_S(..., lam=..., bias=...)` cho mọi
thí nghiệm sau, và vào pipeline cuối cùng ở [mục 5.10](#510--cell-chốt-ghi-đè-stage3_grounding-và-run_inference_vlm).

### 5.5 — Mục 2.1: hỏi `bbox_2d` thay vì `point`

```python
# [PROMPT] Hoi bounding box roi lay tam, thay vi hoi point truc tiep.
#
# Ly do: bbox_2d la dinh dang Qwen3-VL duoc huan luyen nang nhat (detection), va
# tam box la gia tri SUY RA -- model khong phai chon truc tiep mot con so, nen it
# co hoi roi vao cac gia tri 'tron' quen thuoc. Them nua, ground truth cua cuoc
# thi CHINH LA tam cua mot bbox tai nan (center_x == (x1+x2)/2, da kiem tra tren
# labels_df), kich thuoc trung binh 0.0952 x 0.1353 -- tuc dung mot vung tiep xuc,
# khong phai hop bao ca hai xe.

GROUNDING_BOX_PROMPT = (
    'This CCTV frame shows a traffic accident. '
    'Find the point of impact -- where the vehicles make contact, or where a '
    'vehicle strikes an object -- and output a TIGHT bounding box around the '
    'damaged/contact area only (not the whole vehicle, not the whole scene).\n'
    'Output ONLY this JSON: {"bbox_2d": [x1, y1, x2, y2]}\n'
    'Coordinates are on a 0-1000 scale: x horizontal (0=left, 1000=right), '
    'y vertical (0=top, 1000=bottom).'
)


def parse_box_center(text):
    for j in _extract_json_all(text):
        box = j.get('bbox_2d') or j.get('bbox') or j.get('box')
        if isinstance(box, (list, tuple)) and box and isinstance(box[0], (list, tuple)):
            box = box[0]                     # model doi khi tra ve list cua list
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            v = [_safe_float(b, np.nan) for b in box[:4]]
            if all(np.isfinite(b) for b in v):
                x1, y1, x2, y2 = v
                return _to_unit((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return parse_point(text)                 # du phong: co the model van tra ve point


def g_box(vp, t, row=None):
    frames = sample_frames_stamped(vp, 1.0, t, t + 1e-3, limit=1,
                                   max_side=VLM_CFG['grounding_max_side'], burn=False)
    if not frames:
        return None
    return sanitize_point(parse_box_center(
        vlm_generate_oom_safe(GROUNDING_BOX_PROMPT, frames, 96, min_frames=1)))


eval_box = eval_S(g_box, 'v2: bbox_2d -> tam', lam=_lam_mom)
compare_S(eval_box, eval_v1, 'bbox vs point')
```

### 5.6 — Mục 2.2: prompt có điều kiện loại va chạm

```python
# [PROMPT] Grounding co dieu kien loai va cham. Chi phi = 0 lan goi them: loai da
# co san trong cache (type_stage1), va trong pipeline that chi can DOI THU TU --
# goi classify_type_cascade TRUOC stage3 thay vi sau.
#
# Ly do: 'diem va cham' cua rear-end (dau xe sau dap duoi xe truoc) va cua
# t-bone (dau xe nay dap suon xe kia) la hai vi tri hinh hoc khac nhau. Noi cho
# model biet loai la thu hep khong gian tim kiem mien phi.

TYPE_WHERE_HINT = {
    'rear-end':  'This is a rear-end collision: the impact point is where the FRONT of the '
                 'following vehicle meets the REAR of the vehicle ahead.',
    'head-on':   'This is a head-on collision: the impact point is between the FRONTS of the '
                 'two vehicles travelling in opposite directions.',
    'sideswipe': 'This is a sideswipe: the impact point is where the SIDES of two roughly '
                 'parallel vehicles touch.',
    't-bone':    'This is a t-bone collision: the impact point is where the FRONT of one '
                 "vehicle meets the SIDE of the other, at roughly a right angle.",
    'single':    'This is a single-vehicle crash: the impact point is where the vehicle meets '
                 'the object it hits (wall, pole, barrier, guardrail).',
}


def g_box_typed(vp, t, row=None):
    hint = TYPE_WHERE_HINT.get((row or {}).get('type_stage1', ''), '')
    prompt = GROUNDING_BOX_PROMPT.replace(
        'This CCTV frame shows a traffic accident. ',
        f'This CCTV frame shows a traffic accident. {hint} ')
    frames = sample_frames_stamped(vp, 1.0, t, t + 1e-3, limit=1,
                                   max_side=VLM_CFG['grounding_max_side'], burn=False)
    if not frames:
        return None
    return sanitize_point(parse_box_center(
        vlm_generate_oom_safe(prompt, frames, 96, min_frames=1)))


eval_typed = eval_S(g_box_typed, 'v3: bbox + hint loai', lam=_lam_mom)
compare_S(eval_typed, eval_box, 'them hint loai')
```

> **Lưu ý về tính trung thực của phép đo:** `type_stage1` là loại **dự đoán**, không phải
> loại thật — đúng như trong pipeline. Đừng thử với `type_gt`, con số sẽ đẹp một cách vô
> nghĩa.

### 5.7 — Mục 2.3 + 2.5: ngữ cảnh 3 khung, và crop quanh toạ độ Stage 1

```python
# [PROMPT] Dua 3 khung (t-0.4, t, t+0.4) vao CUNG MOT lan goi, hoi diem o khung
# GIUA. Diem va cham la mot su kien CHUYEN DONG -- mot khung tinh co the mo ho,
# nhung 3 khung lien tiep cho model thay xe nao dang lao vao dau. Chi phi token
# tang 3x tren mot lan goi Stage 3 (423 -> ~1270 token) nhung so LAN GOI khong
# doi, va Stage 1 dang dung 16 x 448px = ~2300 token nen day khong phai van de.

GROUNDING_3F_PROMPT = (
    'These are 3 consecutive CCTV frames of a traffic accident, in order. '
    'The MIDDLE frame is the moment of impact. '
    'Using the motion visible across the three frames, locate the point of impact '
    'IN THE MIDDLE FRAME and output a tight bounding box around the contact area.\n'
    'Output ONLY this JSON: {"bbox_2d": [x1, y1, x2, y2]}\n'
    'Coordinates are on a 0-1000 scale relative to the MIDDLE frame: '
    'x horizontal (0=left, 1000=right), y vertical (0=top, 1000=bottom).'
)


def g_3frame(vp, t, row=None):
    frames = []
    for dt in (-0.4, 0.0, 0.4):
        tt = max(0.0, t + dt)
        f = sample_frames_stamped(vp, 1.0, tt, tt + 1e-3, limit=1,
                                  max_side=VLM_CFG['grounding_max_side'], burn=False)
        if f:
            frames.append(f[0])
    if not frames:
        return None
    return sanitize_point(parse_box_center(
        vlm_generate_oom_safe(GROUNDING_3F_PROMPT, frames, 96, min_frames=1)))


# ---------------------------------------------------------------------------
# [ZOOM] Crop quanh toa do Stage 1 roi hoi lai. Van CHI 1 lan goi Stage 3:
# Stage 1 da tra ve mot (x, y) tho mien phi, dung no lam tam crop thay vi tra
# them mot lan goi de hoi tho.
#
# Vi sao zoom quan trong: frame 1920x1080 ha ve 768px, hop ground-truth trung
# binh 0.0952 x 0.1353 con 73x58 px, tuc ~2.6 x 2.1 patch cua Qwen-VL (moi patch
# 28x28 px). Crop 1/3 khung roi resize len 768 lam muc tieu to gap 3 lan MA
# KHONG tang so token. Voi clip that 3840x2160 thi loi ich con lon hon.

def _frame_pil(vp, t, max_side):
    f = sample_frames_stamped(vp, 1.0, t, t + 1e-3, limit=1, max_side=max_side, burn=False)
    return f[0][1] if f else None


def _crop_window(cx, cy, frac):
    """Cua so vuong (theo ti le khung) tam (cx, cy), da day vao trong bien."""
    h = frac / 2.0
    x0, x1 = cx - h, cx + h
    y0, y1 = cy - h, cy + h
    if x0 < 0:  x0, x1 = 0.0, frac
    if x1 > 1:  x0, x1 = 1.0 - frac, 1.0
    if y0 < 0:  y0, y1 = 0.0, frac
    if y1 > 1:  y0, y1 = 1.0 - frac, 1.0
    return max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)


def _ground_on_crop(img, x0, y0, x1, y1, prompt, max_side):
    """Grounding tren anh da crop, roi ANH XA NGUOC ve he toa do khung goc."""
    W, H = img.size
    box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return None
    crop = img.crop(box)
    if max(crop.size) < max_side:                      # phong to de "chi" do phan giai
        sc = max_side / max(crop.size)
        crop = crop.resize((int(crop.size[0] * sc), int(crop.size[1] * sc)))
    pt = parse_box_center(vlm_generate_oom_safe(prompt, [(0.0, crop)], 96, min_frames=1))
    if pt is None:
        return None
    u, v = pt
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    # ANH XA NGUOC -- day la cho de sai am tham nhat trong ca huong nay
    return (x0 + u * (x1 - x0), y0 + v * (y1 - y0))


ZOOM_FRAC = 0.40        # canh cua so crop, theo ti le canh khung


def g_zoom_stage1(vp, t, row=None):
    """1 lan goi: crop quanh toa do Stage 1, hoi tren anh crop."""
    img = _frame_pil(vp, t, VLM_CFG['grounding_max_side'])
    if img is None:
        return None
    cx = float((row or {}).get('x_stage1', 0.5))
    cy = float((row or {}).get('y_stage1', 0.5))
    x0, y0, x1, y1 = _crop_window(cx, cy, ZOOM_FRAC)
    return sanitize_point(_ground_on_crop(img, x0, y0, x1, y1, GROUNDING_BOX_PROMPT,
                                          VLM_CFG['grounding_max_side']))


# KIEM CHUNG BAT BUOC truoc khi tin bat ky con so nao tu 2 ham zoom: crop toan
# khung phai cho ket qua TRUNG voi ban khong crop. Neu lech, phep anh xa nguoc
# dang sai va moi so do sau do vo nghia.
def _selftest_crop_mapping():
    r = rows_s12[0]
    vp, t = pathlib.Path(r['path']), float(r['t_final'])
    img = _frame_pil(vp, t, VLM_CFG['grounding_max_side'])
    full = _ground_on_crop(img, 0.0, 0.0, 1.0, 1.0, GROUNDING_BOX_PROMPT,
                           VLM_CFG['grounding_max_side'])
    direct = g_box(vp, t, r)
    print(f'[SELFTEST] crop toan khung {full}  vs  khong crop {direct}')
    if full and direct:
        d = np.hypot(full[0] - direct[0], full[1] - direct[1])
        print(f'[SELFTEST] khoang cach {d:.4f}  -> '
              f"{'OK' if d < 0.02 else 'SAI ANH XA -- dung lai va sua truoc khi do'}")


_selftest_crop_mapping()

eval_3f   = eval_S(g_3frame,      'v4: 3 khung 1 lan goi', lam=_lam_mom)
eval_zs1  = eval_S(g_zoom_stage1, 'v5: zoom quanh stage1', lam=_lam_mom)
compare_S(eval_3f,  eval_box, '3 khung')
compare_S(eval_zs1, eval_box, 'zoom stage1')
```

### 5.8 — Mục 3.1: coarse-to-fine đầy đủ (thêm 1 lần gọi)

```python
# [ZOOM] Coarse-to-fine day du: hoi tho tren toan khung -> crop quanh cau tra
# loi tho -> hoi lai -> anh xa nguoc. Day dung la ky thuat vua dua T tu 0.4035
# len 0.4385 (cell 111), ap sang truc khong gian. Chi phi: +1 lan goi
# (~+1.4 h cho ca 2027 clip).
#
# Chot an toan: neu buoc fine tra ve mot diem cach buoc coarse qua xa (> nua
# cua so crop) thi giu buoc coarse -- day la dau hieu buoc fine bat sang mot xe
# khac chu khong phai tinh chinh.

FINE_MAX_JUMP = 0.5 * ZOOM_FRAC


def g_coarse2fine(vp, t, row=None):
    coarse = g_box(vp, t, row)
    if coarse is None:
        return None
    img = _frame_pil(vp, t, VLM_CFG['grounding_max_side'])
    if img is None:
        return coarse
    x0, y0, x1, y1 = _crop_window(coarse[0], coarse[1], ZOOM_FRAC)
    fine = _ground_on_crop(img, x0, y0, x1, y1, GROUNDING_BOX_PROMPT,
                           VLM_CFG['grounding_max_side'])
    fine = sanitize_point(fine)
    if fine is None:
        return coarse
    if np.hypot(fine[0] - coarse[0], fine[1] - coarse[1]) > FINE_MAX_JUMP:
        return coarse
    return fine


eval_c2f = eval_S(g_coarse2fine, 'v6: coarse-to-fine', lam=_lam_mom)
compare_S(eval_c2f, eval_box, 'coarse-to-fine')
```

### 5.9 — Mục 3.2: bỏ phiếu nhiều khung + shrinkage thích ứng

```python
# [VOTE] Bo phieu tren nhieu khung quanh t_final, lay TRUNG VI, va tra ve them
# DO TAN de shrinkage thich ung.
#
# Vi sao median chu khong mean: mot lan grounding truot han sang xe khac keo mean
# di rat xa, median thi khong.
#
# Vi sao do tan co gia tri rieng: no la mot uoc luong nhieu KHONG can nhan. Ba
# lan goi dong y -> tin (lam gan 1). Ba lan goi ra ba noi -> khong tin (co manh
# ve prior). Day chinh la lam thich ung o Phan 3.3, ap theo tung video thay vi
# mot lam chung.
#
# Luu y ve ky vong: 3 lan goi tren 3 frame lien ke voi greedy decode co sai so
# TUONG QUAN CAO (rho ~ 0.6-0.8), nen giam phuong sai thuc te chi khoang 0.8x
# chu khong phai 0.33x nhu truong hop doc lap. Dung ky vong +0.2.

VOTE_OFFSETS = (-0.25, 0.0, 0.25)


def g_vote(vp, t, row=None, base_fn=None):
    base_fn = base_fn or g_box
    pts = []
    for dt in VOTE_OFFSETS:
        p = base_fn(vp, max(0.0, t + dt), row)
        if p is not None:
            pts.append(p)
    if not pts:
        return None
    a = np.asarray(pts, dtype=float)
    med = (float(np.median(a[:, 0])), float(np.median(a[:, 1])))
    spread = float(np.median(np.hypot(a[:, 0] - med[0], a[:, 1] - med[1]))) if len(a) > 1 else 0.0
    return (med[0], med[1], spread)


eval_vote = eval_S(g_vote, 'v7: vote 3 khung', lam=_lam_mom)
compare_S(eval_vote, eval_box, 'vote 3 khung')

# Shrinkage thich ung: co manh hon khi 3 lan goi khong dong y
print('\n[SWEEP] shrinkage thich ung theo do tan')
for _gate in (0.05, 0.08, 0.12):
    for _lo in (0.5, 0.4, 0.3):
        _df = eval_S(g_vote, f'gate={_gate} lam_lo={_lo}', lam=_lam_mom,
                     lam_lo=(_lo, _lo), spread_gate=_gate, verbose=False)
        _n = int((_df['spread'] > _gate).sum())
        print(f'   gate={_gate:.2f} lam_lo={_lo:.1f}  S={_df["S"].mean():.4f}  '
              f'({_n}/{len(_df)} video bi co manh)')
```

> Vì `g_vote` gọi lại `base_fn` 3 lần, để tránh trả GPU 3 lần cho mỗi sweep hãy **cache
> điểm thô theo `(stem, dt)`** vào một dict rồi sweep offline. Với 20 video × 3 offset thì
> một lần chạy là đủ cho toàn bộ bảng sweep phía sau.

### 5.10 — Cell chốt: ghi đè `stage3_grounding` và `run_inference_vlm`

Chỉ chạy sau khi đã chọn xong biến thể. Ví dụ với cấu hình `bbox + hint loại + zoom
Stage 1 + shrinkage`:

```python
# [PATCH v3] Ghi de stage3_grounding + run_inference_vlm voi cau hinh S da chot.
# KHONG xoa cell nao -- Python dung dinh nghia moi nhat.
#
# Cau hinh chot (dien lai theo so DO DUOC, dung theo mau nay):
#   parser cung + chan suy bien      S 0.xxxx -> 0.xxxx
#   bbox_2d thay point               S 0.xxxx -> 0.xxxx
#   hint loai va cham                S 0.xxxx -> 0.xxxx
#   zoom quanh toa do stage1         S 0.xxxx -> 0.xxxx
#   shrinkage lam=(0.xx, 0.xx)       S 0.xxxx -> 0.xxxx
# So lan goi Stage 3 moi video: 1 (khong doi so voi ban goc)

S_LAMBDA = (0.00, 0.00)     # <-- dien lam da chot
S_BIAS   = (0.0, 0.0)       # <-- dien bias da chot (0.0 neu khong du bang chung)


def _apply_posthoc(pt):
    if pt is None:
        return None
    mu_x, mu_y = CONST['center_x'], CONST['center_y']
    return (_clip01(mu_x + S_LAMBDA[0] * (pt[0] + S_BIAS[0] - mu_x)),
            _clip01(mu_y + S_LAMBDA[1] * (pt[1] + S_BIAS[1] - mu_y)))


def stage3_grounding(video_path, t_final, type_hint=None, x0y0=None):
    """Ban da chot cho metric S. Tra None khi khong chac -- caller rot ve CONST,
    KHONG rot ve toa do Stage 1 (da do: yeu hon) va KHONG rot ve optical-flow
    (S=0.118, duoi hang so)."""
    if not VLM_CFG['use_grounding']:
        return None
    row = {'type_stage1': type_hint or '',
           'x_stage1': (x0y0 or (0.5, 0.5))[0],
           'y_stage1': (x0y0 or (0.5, 0.5))[1]}
    pt = g_zoom_stage1(video_path, t_final, row)      # <-- doi sang bien the da chon
    return _apply_posthoc(pt)


def run_inference_vlm(video_path: pathlib.Path, sub_path: str = None) -> dict:
    """Stage 1 NumPro -> Stage 2 refine frame-so -> type cascade -> Stage 3
    grounding (co dieu kien loai) -> scene rule.

    THU TU DOI: cascade chay TRUOC grounding de Stage 3 biet loai va cham. Khong
    ton them lan goi nao -- chi doi thu tu.
    """
    if not VLM_AVAILABLE:
        raise RuntimeError('run_inference_vlm called with no usable VLM: every row '
                           'would be normalize_prediction defaults, i.e. a constant.')
    cap = cv2.VideoCapture(str(video_path))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = (n / fps) if fps > 0 else 20.0

    scene = SCENE_BY_PATH.get(sub_path or ('videos/' + video_path.name))
    pred = stage1_full_scan(video_path, duration, scene)

    global PARSE_FAIL_STREAK
    if pred.get('_parsed'):
        PARSE_FAIL_STREAK = 0
    else:
        PARSE_FAIL_STREAK += 1
        if PARSE_FAIL_STREAK >= PARSE_FAIL_ABORT:
            raise RuntimeError(
                f'Stage 1 failed to parse JSON on {PARSE_FAIL_STREAK} consecutive clips. '
                'The predictions being written are normalize_prediction defaults. '
                'Inspect the raw VLM output before continuing.')

    t_final = stage2_time_refine_numbered(video_path, pred['accident_time'], duration)

    col_type = classify_type_cascade(video_path, t_final, pred['type'])
    col_type = apply_scene_type_postfix(col_type, scene)

    pt = stage3_grounding(video_path, t_final, type_hint=col_type,
                          x0y0=(pred['center_x'], pred['center_y']))
    if pt is None:
        pt = (CONST['center_x'], CONST['center_y'])     # fallback = prior, khong phai stage1

    return {'path': str(video_path), 'accident_time': t_final,
            'center_x': pt[0], 'center_y': pt[1],
            'type': col_type, 'scene_layout': scene}


print('[STATUS] stage3_grounding + run_inference_vlm da chot cho metric S')
```

### 5.11 — Cell xác nhận cuối: đo T, S, C CÙNG LÚC

**Bắt buộc trước khi chạy full.** Đây là lỗ hổng lớn nhất hiện tại của notebook — chưa bao
giờ có một lần chạy đo cả ba trên cùng một pipeline.

```python
# [EVAL] Do T, S, C cung luc tren ban chot -- 20 video, ~20 phut.
_rows, _t0 = [], time.time()
for _i, vp in enumerate(diverse_videos, 1):
    _rows.append(run_inference_vlm(vp))
    print(f'[{_i:2d}/20] {vp.name:42s} '
          f"t={_rows[-1]['accident_time']:6.2f} "
          f"xy=({_rows[-1]['center_x']:.3f},{_rows[-1]['center_y']:.3f}) "
          f"{_rows[-1]['type']:10s} ({(time.time()-_t0)/_i:.1f}s/video)")

eval_final_s = score_predictions(pd.DataFrame(_rows), diverse_labels_df)
print(f"\n[CHOT]  T={eval_final_s['T'].mean():.4f}  S={eval_final_s['S'].mean():.4f}  "
      f"C={eval_final_s['C'].mean():.4f}  ACCS={accident_score(eval_final_s):.4f}")
print(f"[FLOOR] T={const_eval['T'].mean():.4f}  S={const_eval['S'].mean():.4f}  "
      f"C={const_eval['C'].mean():.4f}  ACCS={accident_score(const_eval):.4f}")
for _c in ('T', 'S', 'C'):
    _a, _b = eval_final_s[_c].mean(), const_eval[_c].mean()
    print(f"  {_c}: {_a:.4f} vs hang so {_b:.4f}  {'THANG' if _a > _b else 'THUA'}")
print(f"\n[TIMING] {(time.time()-_t0)/20:.1f}s/video -> "
      f"{(time.time()-_t0)/20*len(real_videos)/3600:.1f}h cho {len(real_videos)} clip that")
display(pd.crosstab(eval_final_s['type_gt'], eval_final_s['type_pred']))
```

---

## Phần 6 — Quy trình đo lường (đọc trước khi tin bất kỳ con số nào)

### 6.1 Bốn quy tắc bắt buộc

1. **Luôn so với baseline MỚI**, không so với 0.3987. Con số 0.3987 thuộc một `t_final`
   khác và một pipeline khác.
2. **Đọc khoảng tin cậy, không đọc chữ số.** Harness ở mục 5.1 in bootstrap 95% CI. Với
   n = 20, CI của S rộng khoảng ±0.10. **Chênh lệch dưới +0.05 trên n=20 là nhiễu.**
3. **Xem bảng delta từng video** (`compare_S`). Một video từ 0.02 lên 0.95 đủ đẩy trung
   bình +0.046 mà không chứng minh được gì. Yêu cầu: cải thiện xuất hiện ở **≥ 60% video**.
4. **Kiểm tra ACCS tổng không giảm.** Nếu bạn đổi thứ tự cascade/grounding và vô tình làm
   C giảm, điểm tổng có thể tụt dù S tăng.

### 6.2 Mở rộng tập đánh giá cho S lên 60 video

Đây là khuyến nghị mạnh. Stage 3 chỉ 1 frame nên **rất rẻ** — có thể đánh giá trên 60
video mà tổng chi phí vẫn thấp hơn một lần chạy pipeline đầy đủ trên 20 video. n = 60 giảm
sai số chuẩn xuống khoảng 58%, biến "+0.04 có thể là nhiễu" thành một kết luận.

Mẹo: để đo **riêng Stage 3** (không cần Stage 1/2), dùng **`accident_time` thật** làm
`t_final`. Vòng lặp chỉ tốn 1 lần gọi VLM mỗi video.

```python
# [EVAL] Tap danh gia S mo rong -- 12 video moi loai, dung t THAT nen KHONG can
# chay Stage 1/2. Do RIENG chat luong Stage 3 (S_ORACLE), n=60 thay vi 20.
S_EVAL_PER_TYPE = 12
_rows60 = []
for _ct in COLLISION_TYPES:
    _sub = labels_clean[labels_clean['type'] == _ct].sample(
        min(S_EVAL_PER_TYPE, (labels_clean['type'] == _ct).sum()), random_state=SEED + 1)
    for _, _r in _sub.iterrows():
        _rows60.append({'path': str(_r['abs_video_path']), 'duration': float(_r['duration']),
                        't_final': float(_r['accident_time']),   # dung t THAT
                        't_gt': float(_r['accident_time']),
                        'x_stage1': 0.5, 'y_stage1': 0.5, 'type_stage1': _ct,
                        'x_gt': float(_r['center_x']), 'y_gt': float(_r['center_y']),
                        'type_gt': _ct})
print(f'[STATUS] tap S mo rong: {len(_rows60)} video')

eval_60_base = eval_S(g_base, 'goc (n=60, t that)', rows=_rows60)
eval_60_new  = eval_S(g_box,  'bbox (n=60, t that)', rows=_rows60, lam=_lam_mom)
compare_S(eval_60_new, eval_60_base, 'n=60')
```

Lưu ý: `type_stage1` ở đây bằng loại **thật** — chỉ dùng cho *tập mở rộng này*, và **không
được** dùng nó để kết luận về mục 2.2 (hint loại). Với mục 2.2, chỉ tin số đo trên
`rows_s12` (loại dự đoán).

### 6.3 Hai chế độ đo, và cách đọc khoảng cách giữa chúng

| Chế độ | `t_final` | Trả lời câu hỏi |
|---|---|---|
| `S_ORACLE` | thời điểm va chạm **thật** | Stage 3 giỏi đến đâu, nếu T hoàn hảo? |
| `S_PIPELINE` | `t_final` dự đoán | Điểm thật sự lên leaderboard |

`S_ORACLE − S_PIPELINE` chính là **phần điểm S đang bị T giữ lại**. Nếu khoảng cách này
lớn (> 0.08), việc tiếp tục cải thiện T (hoặc thêm bỏ phiếu nhiều khung ở mục 3.2 để bù
cho `t_final` lệch) có lợi hơn là tối ưu prompt.

---

## Phần 7 — Ngân sách GPU và cấu hình chốt

### 7.1 Chi phí

Từ dry-run thật (cell 97): **49.83 s/video → 28.06 h cho 2027 clip**.

| Cấu hình | s/video | Tổng | Số session 12 h |
|---|---|---|---|
| Hiện tại | 49.8 | **28.1 h** | 3 |
| +1 lần gọi Stage 3 | ~52.3 | 29.5 h | 3 |
| +2 lần gọi | ~54.8 | 30.9 h | 3 |
| +3 lần gọi | ~57.3 | 32.3 h | 3 |

Quota Kaggle: **30 h GPU/tuần**, mỗi session ≤ 12 h. Nghĩa là **một lượt nộp đầy đủ đã ăn
gần hết quota tuần**. Cell 115 đã có checkpoint mỗi 5 video + `SESSION_TIME_BUDGET_HOURS`
— giữ nguyên, đó là phần làm đúng.

### 7.2 Cấu hình tôi khuyến nghị nhắm tới

| Thành phần | Chọn | Lần gọi thêm |
|---|---|---|
| Parser + chặn suy biến | bắt buộc | 0 |
| Shrinkage λ (theo moment, xác nhận bằng sweep) | bắt buộc | 0 |
| Hình thức hỏi | `bbox_2d` nếu thắng `point` | 0 |
| Hint loại va chạm | có, đảo thứ tự cascade lên trước | 0 |
| Zoom | `g_zoom_stage1` (crop quanh toạ độ Stage 1) | 0 |
| Bỏ phiếu nhiều khung | **chỉ nếu** `S_ORACLE − S_PIPELINE` còn lớn | +2 |
| Fallback | `CONST`, không bao giờ về optical-flow/OWLv2 | 0 |

Cấu hình này giữ **đúng 1 lần gọi Stage 3 mỗi video** — tổng thời gian không đổi so với
hiện tại (28.1 h) — mà theo ước tính ở Phần 3.4 vẫn đạt vùng **S ≈ 0.50–0.55**.

### 7.3 Nếu vẫn muốn thêm ngân sách

Ưu tiên `g_coarse2fine` (+1 gọi) hơn `g_vote` (+2 gọi): nó nhắm vào nguồn mất điểm D (độ
phân giải), có cơ chế nhân quả rõ và đã có tiền lệ thành công trên trục thời gian; còn
`g_vote` phụ thuộc vào giả định các lần gọi ít tương quan, điều mà greedy decode làm suy
yếu.

---

## Phần 8 — Một lưu ý ngoài phạm vi: C đang là mắt yếu hơn S

Vì điểm cuối là trung bình điều hoà, đạo hàm theo từng thành phần là
`∂ACCS/∂X = ACCS²/(3X²)` — thành phần nhỏ nhất có đòn bẩy lớn nhất. Ở điểm hiện tại
(T=0.4385, S=0.3987, C=0.25, ACCS=0.3413):

| Thành phần | Giá trị | ∂ACCS/∂X | +0.10 đổi được |
|---|---|---|---|
| C | **0.25** | **0.621** | **+0.062** |
| S | 0.3987 | 0.244 | +0.024 |
| T | 0.4385 | 0.202 | +0.020 |

**C = 0.25 chỉ hơn mức ngẫu nhiên (0.20) một chút**, và ma trận nhầm lẫn ở cell 94 cho
thấy toàn bộ 20 clip chỉ được gán 2 nhãn (`t-bone` 17, `sideswipe` 3) — kể cả 4 clip
`rear-end` và 4 clip `single`:

```
type_pred   sideswipe  t-bone
head-on             1       3
rear-end            0       4
sideswipe           1       3
single              1       3
t-bone              0       4
```

Đây là **collapse**, không phải "phân loại kém". `notebooks/02_type_C.ipynb` đã đo được
prior content-free của model: `t-bone 0.48`, `sideswipe 0.24`, `single 0.16`,
`rear-end 0.08`, `head-on 0.04` — model thiên vị `t-bone` gấp 12 lần `head-on` **ngay cả
khi không có thông tin hình ảnh**. Cách sửa có tiền lệ (Zhao et al., ICML 2021 — *Calibrate
Before Use*) là **hiệu chỉnh xác suất bằng prior content-free** đó, chứ không phải thêm
prompt.

Tôi để mục này ngoài lộ trình vì bạn hỏi về S. Nhưng nếu mục tiêu là điểm leaderboard,
**làm C trước S sẽ đổi được nhiều điểm hơn với ít GPU hơn** (calibration là hậu xử lý, gần
như miễn phí). Nói một câu nếu bạn muốn tôi viết kế hoạch riêng cho C.

---

## Phần 9 — Bẫy đã có tiền lệ trong repo, đừng lặp lại

### 9.1 Fallback về model yếu hơn hằng số

Repo đã đo: OWLv2 + optical flow đạt **S = 0.156**, optical-flow centroid **0.1177**, cả
hai **thua** hằng số. Khi grounding trả `None`, chỉ được rơi về **`CONST`** hoặc về toạ độ
Stage 1 — và phải **đo** xem cái nào tốt hơn (harness có tham số `fallback` cho đúng việc
này). Đừng rơi về một model khác.

### 9.2 Overfit λ và bias trên n=20

Shrinkage là công cụ mạnh nhưng λ tối ưu tìm bằng grid-search trên 20 video là **1 tham số
fit trên 20 điểm** — dễ overfit. Ba lớp bảo vệ, hãy dùng cả ba:

1. Ưu tiên **λ theo phương pháp moment** (`Var(nhãn)/Var(dự đoán)`) làm giá trị chính, dùng
   sweep chỉ để **đối chiếu**. Nếu hai con số lệch xa, tin bản moment.
2. `Var(dự đoán)` có thể tính trên **chính 2027 dự đoán của tập test**, không cần nhãn. Đây
   là cách chắc chắn nhất để λ khớp với phân bố thật thay vì với CARLA.
3. Với bias, chỉ áp dụng khi `|mean(dx)| > sem(dx)`. Dưới ngưỡng đó thì bạn đang fit nhiễu.

### 9.3 Set-of-Mark có thể phản tác dụng

`02_type_C` đã đo được: **VLM collapse khi phải chọn từ một tập giá trị neo rời rạc, kể cả
khi neo là số.** Set-of-Mark (mục 3.3) chính là một tập rời rạc như vậy. Nó có thể thắng
lớn (biến hồi quy thành phân loại, xoá sạch lỗi `(0,0)`) hoặc thua đậm (model luôn chọn
mark số 1, hoặc chọn xe to nhất). **Phải đo trên đủ 5 loại trước khi tin**, và kiểm tra
phân bố mark được chọn có suy biến không.

### 9.4 Tinh chỉnh trên tập lệch

Một bản trước đã tune trên tập chỉ có `head-on`, đạt 0.80 ở đó và 0.10 trên tập đa dạng.
Luôn đánh giá trên đủ 5 loại. Bảng `compare_S` in cột `type_gt` cho đúng mục đích này —
nếu cải thiện chỉ đến từ một loại, đó không phải cải thiện.

### 9.5 Hai chi tiết kỹ thuật dễ sai âm thầm

- **Ánh xạ ngược toạ độ crop.** Sai dấu hoặc sai thứ tự `x0 + u*(x1-x0)` cho ra con số hợp
  lệ nhưng vô nghĩa. Hàm `_selftest_crop_mapping()` ở mục 5.7 là bắt buộc, không phải tuỳ
  chọn.
- **`SCENE_BY_PATH.get('videos/' + vp.name)` luôn trả `None` trên tập calibration** — nó
  được dựng từ `test_metadata.csv` (2027 clip thật), còn video calibration là synthetic.
  Nghĩa là **mọi số đo trên calibration đều KHÔNG có scene hint**. Điểm trên tập thật đáng
  lẽ *cao hơn* một chút. Đây không phải bug cần sửa, nhưng đừng ngạc nhiên khi thấy
  `apply_scene_type_postfix` không bao giờ kích hoạt lúc calibrate.

---

## Phần 10 — Bảng theo dõi kết quả (điền khi chạy)

Copy bảng này vào một markdown cell **đầu** notebook và cập nhật sau mỗi thí nghiệm — để
người đọc (và chính bạn tuần sau) không phải chạy lại mới biết số.

```markdown
## Kết quả metric S (cập nhật lần cuối: ____)

Baseline cần vượt:
| Mốc | S | Ghi chú |
|---|---|---|
| Constant floor (0.51, 0.51) | 0.2159 | trên đúng 20 video calibration |
| Báo cáo cũ ở cell 94 | 0.3987 | đo với t_final CŨ (sai TB 3.10 s) — không dùng để so |
| **Baseline mới (t_final NumPro+refine)** | ______ | ← mốc thật |
| S_ORACLE (grounding tại t THẬT) | ______ | trần của Stage 3 hiện tại |

Thí nghiệm:
| # | Biến thể | S | Δ vs baseline mới | 95% CI | Số video tốt lên | Lần gọi thêm | Giữ? |
|---|---|---|---|---|---|---|---|
| v1 | parser + chặn suy biến | ____ | ____ | ____ | __/20 | 0 | |
| — | + shrinkage λ=(__,__) | ____ | ____ | ____ | __/20 | 0 | |
| v2 | bbox_2d thay point | ____ | ____ | ____ | __/20 | 0 | |
| v3 | + hint loại va chạm | ____ | ____ | ____ | __/20 | 0 | |
| v4 | 3 khung trong 1 lần gọi | ____ | ____ | ____ | __/20 | 0 | |
| v5 | zoom quanh toạ độ Stage 1 | ____ | ____ | ____ | __/20 | 0 | |
| v6 | coarse-to-fine | ____ | ____ | ____ | __/20 | +1 | |
| v7 | vote 3 khung + λ thích ứng | ____ | ____ | ____ | __/20 | +2 | |

Bản chốt (đo T/S/C CÙNG LÚC, cell 5.11):
| | T | S | C | ACCS | s/video | ETA 2027 clip |
|---|---|---|---|---|---|---|
| Constant floor | 0.3800 | 0.2159 | 0.2000 | 0.2446 | — | — |
| Bản chốt | ____ | ____ | ____ | ____ | ____ | ____ h |
```

---

## Phụ lục — Tra cứu nhanh

### A. Cell nào làm gì (notebook hiện tại, 116 cell)

| Cần gì | Cell |
|---|---|
| `SIGMA_X`, `SIGMA_Y`, `CONST`, `fit_constant_predictor` | 58 |
| Load Qwen3-VL, `VLM_CFG` | 66 |
| Prompts, `sample_frames_stamped`, `vlm_generate`, `vlm_generate_oom_safe`, `_extract_json` | 67 |
| Canary | 72 |
| **`stage3_grounding` gốc**, `normalize_prediction`, `_clip01`, `_safe_float` | **73** |
| `classify_type_cascade` | 76 |
| `temporal_score`, `spatial_score`, `accident_score`, `score_predictions` | 81 |
| `diverse_videos`, `diverse_labels_df` (20 video, seed 42) | 82 |
| `const_eval` (constant floor) | 84 |
| `eval_vlm` — nguồn của S=0.3987 | 94 |
| Submission gốc (checkpoint 25 video) | 97–98 |
| Ablation 3 lớp temporal (T từng lớp) | 105 |
| NumPro A/B | 107 |
| **`stage1_full_scan` bản NumPro, `whole_limit=16`** | **109** |
| **`stage2_time_refine_numbered`** | **111** |
| **`run_inference_vlm` bản chốt** | **113** |
| Submission v2 (checkpoint 5 video + dừng sớm) | 115 |

### B. Hàm/biến của notebook mà code trong tài liệu này dùng

`sample_frames_stamped`, `vlm_generate_oom_safe`, `_extract_json`, `_safe_float`,
`_clip01`, `VLM_CFG`, `GROUNDING_PROMPT`, `spatial_score`, `score_predictions`,
`accident_score`, `diverse_videos`, `diverse_labels_df`, `labels_clean`, `CONST`,
`SIGMA_X`, `SIGMA_Y`, `COLLISION_TYPES`, `SEED`, `OUTPUT_DIR`, `SCENE_BY_PATH`,
`classify_type_cascade`, `apply_scene_type_postfix`, `stage1_full_scan`,
`stage2_time_refine_numbered`, `PARSE_FAIL_STREAK`, `PARSE_FAIL_ABORT`, `real_videos`,
`yolo_model`, `VEHICLE_CLASS_IDS`, `YOLO_NMS_IOU`, `const_eval`.

### C. Tham khảo

- Kaggle: **ACCIDENT @ CVPR 2026** — AUTOPILOT Workshop
- Writeup hạng 1: **arXiv:2605.29325** — Qwen3-VL, 3 lần gọi/clip, không train, 0.57080.
  Tách Stage 3 grounding riêng là cải tiến đơn lẻ lớn nhất của họ: **+0.09356**. Một lần
  gọi đơn lẻ ở 768 px/2 fps = 0.42238. Detection chỉ còn giá trị như bước snap hậu xử lý
  (+0.0005/+0.0013)
- **NumPro** — Wu et al., *Number it: Temporal Grounding Videos like Flipping Manga*,
  CVPR 2025, arXiv:2411.10332 (cơ sở của cell 107/109/111 — đã đo được T 0.2386 → 0.4385)
- **Set-of-Mark** — Yang et al., *Set-of-Mark Prompting Unleashes Extraordinary Visual
  Grounding in GPT-4V*, arXiv:2310.11441 (cơ sở của mục 3.3)
- **Calibrate Before Use** — Zhao et al., ICML 2021 (phương pháp cho mục C ở Phần 8)
- Wiener / **James–Stein shrinkage** — cơ sở toán của mục 1.3; ở đây dùng dạng đóng
  `λ* = τ²/(τ² + s²)` và estimator moment `λ* = Var(gt)/Var(pred)`
- `PHAN_TICH_ACCIDENT_CVPR.md`, `HUONG_DAN_03_SPATIAL_S.md` — hai tài liệu phân tích trước
  trong repo

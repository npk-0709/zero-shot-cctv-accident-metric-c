# BÁO CÁO ĐỀ XUẤT — Nâng metric S (Spatial Grounding) của pipeline Qwen3-VL


| **Đính kèm 1** | [`DEMO_TANG_METRIC_S.ipynb`](DEMO_TANG_METRIC_S.ipynb) — toàn bộ code chốt + output |
| **Đính kèm 2** | [`KE_HOACH_TANG_METRIC_S.md`](KE_HOACH_TANG_METRIC_S.md) — phân tích kỹ thuật đầy đủ (chẩn đoán, toán, mô phỏng) |

---

## 1. Tóm tắt 30 giây

- **S hiện báo cáo 0.3987 là con số đo trong điều kiện sai:** nó được đo khi `t_final` (thời
  điểm dùng để chọn frame grounding) còn lệch trung bình **3.10 s** so với va chạm thật. Sau
  nâng cấp NumPro cho T (đã làm xong, T 0.2724 → 0.4385), sai số chỉ còn **2.15 s** — nhưng
  **S chưa được đo lại**.
- Đề xuất **5 thay đổi cho Stage 3**, tất cả **không thêm lần gọi model nào** — thời gian chạy
  full 2027 clip **giữ nguyên ~28 giờ**: (1) đo lại baseline, (2) sửa parser + chặn toạ độ suy
  biến `(0,0)`, (3) co dự đoán về prior (shrinkage λ), (4) hỏi `bbox_2d` thay vì `point` +
  hint loại va chạm, (5) crop-zoom quanh toạ độ Stage 1.
- **Kỳ vọng (ước tính từ mô phỏng đã hiệu chỉnh): S 0.40 → 0.50–0.55; ACCS 0.2947 → ~0.37.**
- **Chi phí một lần để đo đạc và chốt: ~60–75 phút GPU** trên tập calibration 20 video.
- **Rủi ro thấp:** chỉ THÊM cell vào cuối notebook, không sửa/xoá cell nào; mỗi thay đổi có
  tiêu chí giữ/loại bằng số đo; rollback = bỏ 1 cell.

**Cần trưởng nhóm:** duyệt chạy phiên đo đạc (mục 6), và trả lời 3 câu hỏi ở mục 9.

---

## 2. Quy ước đọc số liệu trong báo cáo này

| Nhãn | Nghĩa |
|---|---|
| **[ĐO]** | Số thật, đã có trong output của notebook chính |
| **[ƯỚC TÍNH]** | Từ mô phỏng Monte-Carlo được hiệu chỉnh để tái tạo đúng các số [ĐO] (chi tiết: Kế hoạch, Phần 3.4) |
| **[MẪU]** | Output minh hoạ trong notebook demo — sẽ bị thay bằng số thật sau khi chạy |

Mọi con số trong `DEMO_TANG_METRIC_S.ipynb` đều là **[MẪU]** và dòng đầu mỗi output đều ghi rõ
điều đó.

---

## 3. Hiện trạng và căn cứ (toàn bộ là số [ĐO])

### 3.1 Điểm hiện tại trên tập calibration 20 video

| | T | S | C | ACCS |
|---|---|---|---|---|
| Constant floor (đoán 1 bộ số cố định) | 0.3800 | 0.2159 | 0.2000 | 0.2446 |
| Pipeline Qwen3-VL — lần đo gần nhất (cell 94) | 0.2724 | **0.3987** | 0.2500 | 0.2947 |
| Sau nâng cấp T bằng NumPro (cell 111) | **0.4385** | *chưa đo lại* | *chưa đo lại* | *chưa đo* |

### 3.2 Bốn nguồn mất điểm của S đã xác định được

**A. Grounding trên frame sai (lớn nhất — đã được sửa gián tiếp, chỉ còn thiếu phép đo).**
Stage 3 nhìn đúng **một frame tại `t_final`**. Con số S=0.3987 được đo khi `t_final` sai trung
bình 3.10 s — tức model thường bị hỏi "va chạm ở đâu" trên frame **chưa có hoặc đã tan va chạm**.
Sau nâng cấp NumPro (đã merge, cell 109/111/113):

| | t_final cũ (lúc đo S=0.3987) | t_final mới (hiện tại) |
|---|---|---|
| Sai số trung bình so với va chạm thật | 3.10 s | **2.15 s** |
| Số clip có frame grounding trong ±1 s | 5/20 | **8/20** |
| Ví dụ `Town03_head-on_night_40` | lệch 9.2 s | lệch 0.4 s |

**B. Toạ độ suy biến `(0.000, 0.000)` — 2/20 clip calibration và ≥1/5 clip thật (dry-run).**
Mỗi clip như vậy nhận S = 0 thay vì ~0.22 nếu rơi về hằng số. Đã tìm ra 3 đường sinh ra nó
trong code parse (regex bắt bừa cặp số trong ngoặc; `_extract_json` greedy hỏng khi có 2 khối
JSON; quy đổi thang [0,1000] xét từng số riêng lẻ).

**C. Dự đoán tán rộng hơn nhãn thật (over-dispersion).** Độ lệch chuẩn toạ độ dự đoán
(0.235, 0.225) so với nhãn (0.130, 0.181). Với hàm điểm Gaussian, đây là bằng chứng trực tiếp
rằng **co dự đoán về prior sẽ tăng điểm kỳ vọng** — có công thức đóng, không phải tinh chỉnh mò.

**D. Độ phân giải hiệu dụng quá thấp.** Hộp ground-truth trung bình 0.095 × 0.135 khung hình;
ở 768 px chỉ còn **≈ 2×2 patch** của Qwen3-VL (clip thật 3840×2160 còn tệ hơn). Mục tiêu quá
nhỏ để định vị chính xác nếu không phóng to.

---

## 4. Năm thay đổi đề xuất

Tất cả chỉ **THÊM cell** vào cuối notebook (giống cách đã làm với NumPro ở cell 109–113),
không sửa/xoá cell nào. Code đầy đủ + output mẫu của từng thay đổi nằm trong
`DEMO_TANG_METRIC_S.ipynb` (Bước 0–10).

| # | Thay đổi | Nhắm vào | GPU thêm/clip | Kỳ vọng ΔS | Rủi ro |
|---|---|---|---|---|---|
| 1 | **Đo lại baseline S** trên `t_final` mới + đo `S_ORACLE` (grounding tại thời điểm thật) | A | 0 | reset mốc so sánh (dự kiến đã > 0.44) | — |
| 2 | **Parser cứng hơn + chặn suy biến**: sửa 3 lỗi parse; `(0,0)`/`(1,1)` → coi là thất bại → rơi về `CONST`; kẹp toạ độ vào dải phân vị 0.5–99.5% của nhãn | B | 0 | +0.02 → +0.03 | Thấp |
| 3 | **Shrinkage λ về prior**: `q = CONST + λ·(p − CONST)`, λ đặt bằng phương pháp moment `λ = Var(nhãn)/Var(dự đoán)` — ước lượng được **không cần nhãn**, áp lại được trên chính 2027 dự đoán tập test | C | 0 | +0.04 → +0.08 | Trung bình (chống overfit: mục 7) |
| 4 | **Đổi cách hỏi**: `bbox_2d` (thứ Qwen3-VL được train nặng nhất, và GT của cuộc thi chính là tâm bbox tai nạn) + **hint loại va chạm** (đảo `classify_type_cascade` lên trước grounding — chỉ đổi thứ tự, không thêm lần gọi) | B, D | 0 | +0.02 → +0.06 | Thấp |
| 5 | **Zoom quanh toạ độ Stage 1**: crop 40% khung quanh (x, y) mà Stage 1 đã trả về sẵn, phóng to rồi mới hỏi — tăng độ phân giải hiệu dụng ~2.5× mà vẫn đúng 1 lần gọi; kèm **self-test ánh xạ toạ độ bắt buộc** | D | 0 | +0.02 → +0.06 | Trung bình (ánh xạ ngược — đã có self-test) |

**Phương án dự phòng (chỉ khi #5 không đạt):** coarse-to-fine đầy đủ (+1 lần gọi, +1.4 h/full
run). Đã có sẵn code + ngưỡng quyết định trong demo (Bước 8): chỉ giữ nếu Δ ≥ +0.03.

**Các hướng đã cân nhắc nhưng KHÔNG đưa vào đợt này** (lý do trong Kế hoạch, Phần 4/9): bỏ phiếu
3 khung (+2 lần gọi, lợi ích bị hạn chế vì greedy decode làm các lần gọi tương quan cao),
Set-of-Mark với YOLO (rủi ro collapse như đã đo được ở metric C), flip TTA, YOLO snap.

---

## 5. Kết quả kỳ vọng

Kịch bản mẫu trong notebook demo (nhất quán với dải [ƯỚC TÍNH] 0.50–0.55 của mô phỏng):

| Bước | S | Ghi chú |
|---|---|---|
| Constant floor | 0.2159 **[ĐO]** | mốc sàn |
| Báo cáo cũ (cell 94) | 0.3987 **[ĐO]** | đo trên t_final sai — không dùng làm mốc |
| 1. Baseline mới (t_final NumPro) | 0.4421 **[MẪU]** | chỉ nhờ T tốt lên, chưa sửa gì ở Stage 3 |
| 2. + parser + chặn suy biến | 0.4676 **[MẪU]** | |
| 3. + shrinkage λ = (0.562, 1.000) | 0.5031 **[MẪU]** | |
| 4. + bbox_2d + hint loại | 0.5348 **[MẪU]** | |
| 5. + zoom quanh toạ độ Stage 1 | **0.5521 [MẪU]** | cấu hình chốt, vẫn 1 lần gọi |

Tác động lên điểm tổng (trung bình điều hoà của T, S, C):

| | T | S | C | ACCS |
|---|---|---|---|---|
| Lần đo gần nhất **[ĐO]** | 0.2724 | 0.3987 | 0.2500 | **0.2947** |
| Sau đợt này **[ƯỚC TÍNH]** | 0.4385 [ĐO] | ~0.52–0.55 | 0.2500 | **~0.36–0.37** |

> Để đối chiếu: hằng số đạt 0.2446; hệ thắng giải (32B, 3 lần gọi/clip) đạt 0.5708 trên tập thật.

---

## 6. Kế hoạch thực hiện và chi phí

### Phiên đo đạc (1 session Kaggle, tổng ~1.5–2 h trong đó ~60–75 phút GPU tính toán)

| Bước | Việc | Thời gian |
|---|---|---|
| 0 | Chạy notebook chính đến cell 113 (nạp model, định nghĩa hàm) | ~25–35 phút (chủ yếu tải model) |
| 1 | Cache Stage 1+2 cho 20 video calibration (chạy một lần, tất định) | ~15 phút |
| 2 | Baseline mới + `S_ORACLE` + chẩn đoán (demo Bước 1–2) | ~8 phút |
| 3 | v1 parser + shrinkage (demo Bước 3–4) | ~4 phút (sweep λ không tốn GPU) |
| 4 | A/B: bbox → +hint → +zoom (demo Bước 5–7, mỗi biến thể ~3 phút) | ~10 phút |
| 5 | (tuỳ chọn) coarse-to-fine nếu zoom không đạt | ~6 phút |
| 6 | Chốt cấu hình + **đo T/S/C cùng lúc** (demo Bước 9–10) | ~17 phút |

### Chạy full 2027 clip (sau khi nghiệm thu đạt)

- **Không đổi so với hiện tại: ~28 giờ** (49.8 s/video [ĐO] từ dry-run), vì cấu hình chốt giữ
  đúng 1 lần gọi Stage 3/clip. Cần 3 session 12 h; cơ chế checkpoint mỗi 5 video + tự dừng ở
  11 h đã có sẵn (cell 115).
- Quota Kaggle 30 h GPU/tuần → một lượt nộp ăn gần hết quota tuần; phiên đo đạc nên chạy đầu
  tuần, full run cuối tuần hoặc tách tài khoản theo quy ước nhóm.

### Tiêu chí nghiệm thu (điều kiện dừng/đi tiếp)

1. Mỗi thay đổi chỉ được giữ nếu **ΔS ≥ +0.03** so với bước trước **và ≥ 60% video tốt lên**
   (bảng per-video in sẵn trong harness — chống "may mắn 1–2 video" vì n=20 rất nhỏ).
2. Bước 10 (đo T/S/C cùng lúc) phải cho **ACCS ≥ 0.2947** và không thành phần nào thua hằng số
   — nếu không, dừng lại báo nhóm, không chạy full.
3. Self-test ánh xạ crop phải OK (khoảng cách < 0.02) trước khi tin số đo của zoom.
4. Trước khi nộp: check suy biến của submission (số loại phân biệt, std toạ độ) như cell 98
   đang làm.

---

## 7. Rủi ro và biện pháp

| Rủi ro | Mức | Biện pháp |
|---|---|---|
| Overfit λ (shrinkage) vào 20 video CARLA | Trung bình | Dùng λ theo **phương pháp moment** (chỉ cần phương sai, không cần nhãn) thay vì grid-search; ước lượng lại được trên chính 2027 dự đoán tập test; sweep chỉ để đối chiếu. Bias chỉ áp khi \|mean\| > 1 SE. |
| Sai ánh xạ ngược toạ độ khi crop-zoom | Trung bình | Self-test bắt buộc trong code (crop toàn khung phải trùng bản không crop); có chốt chặn "fine nhảy quá xa coarse thì giữ coarse". |
| Hint loại sai (C mới đạt 0.25) kéo grounding lệch | Thấp | Hint viết theo hướng mô tả hình học, không ép buộc; A/B có/không hint bằng số đo; đo bằng loại **dự đoán** đúng như pipeline thật. |
| n=20 quá nhỏ, kết luận nhiễu | Trung bình | Harness in bootstrap 95% CI + bảng per-video; ngưỡng giữ +0.03; nếu hai biến thể sát nhau, có sẵn phương án đo trên tập mở rộng n=60 chỉ cho S (rẻ, 1 lần gọi/video — Kế hoạch, Phần 6.2). |
| Kết quả thật thấp hơn ước tính | — | Ước tính chỉ dùng để xếp ưu tiên; quyết định giữ/loại hoàn toàn bằng số đo thật theo mục 6. Trường hợp xấu nhất vẫn còn +0.02 chắc chắn từ việc chặn suy biến, và baseline mới nhờ T đã tốt lên. |
| Kaggle cắt session giữa chừng | Thấp | Mọi kết quả trung gian cache ra CSV (`rows_stage12_numpro.csv`); bật Persistence "Files only". |

---

## 8. Ghi chú ngoài phạm vi: C mới là thành phần có đòn bẩy lớn nhất

Vì ACCS là trung bình điều hoà, đạo hàm theo từng thành phần tại điểm hiện tại
(T=0.4385, S=0.3987, C=0.25) là: **C: 0.62 — S: 0.24 — T: 0.20**, tức +0.10 cho C đổi được
+0.062 ACCS, gấp ~2.5 lần cùng mức cải thiện ở S. C hiện chỉ hơn mức đoán ngẫu nhiên (0.25 so
với 0.20) và đang collapse về `t-bone`/`sideswipe` trên cả 20 clip [ĐO]. Notebook `02_type_C`
đã đo được prior content-free của model (`t-bone` 0.48 so với `head-on` 0.04) — cách sửa có tiền
lệ (Calibrate Before Use, Zhao et al., ICML 2021) và gần như **miễn phí GPU** (hậu xử lý xác
suất).

Đề xuất: sau khi chốt đợt S này, làm tiếp **đợt C** theo cùng quy trình. Nếu trưởng nhóm đồng ý,
tôi chuẩn bị kế hoạch + demo tương tự.

---

## 9. Xem xét thêm dự án

1. **Duyệt phiên đo đạc** (~1.5–2 h GPU, mục 6) để thay toàn bộ số [MẪU] bằng số [ĐO]?
2. Nếu zoom (thay đổi #5) không đạt ngưỡng, **có cho phép phương án +1 lần gọi/clip**
   (coarse-to-fine, full run 28 h → 29.5 h) hay giữ cứng 1 lần gọi?
3. **Có làm tiếp đợt C** (calibration xác suất, kỳ vọng đòn bẩy ACCS lớn hơn S) ngay sau đợt
   này không?

---
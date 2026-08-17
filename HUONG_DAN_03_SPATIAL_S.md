# Hướng dẫn chạy `notebooks/03_spatial_S.ipynb`

> Tài liệu này phân tích toàn bộ hệ thống `zero-shot-cctv-accident` và hướng dẫn chi tiết
> cách chạy notebook `03_spatial_S.ipynb` — notebook phụ trách metric **S (Spatial
> Grounding)**, tức là dự đoán *toạ độ điểm va chạm* trong khung hình.

---

## TL;DR — 5 điều cần biết trước khi bắt tay

1. **Notebook này KHÔNG chạy được trên máy Windows local của bạn.** Nó cần GPU ≥ 16 GB VRAM
   để load Qwen3-VL-8B (4-bit), cần dataset mount ở `/kaggle/input/...`, và cần internet để
   `pip install` + tải model. Bắt buộc chạy trên **Kaggle** (hoặc Colab, nhưng phải sửa
   đường dẫn dataset).
2. **Notebook hiện tại sẽ CRASH ở cell 4 và cell 9** vì thiếu 3 tên biến/hàm mà core không
   định nghĩa: `eval_core`, `const_eval`, `stage2_time_refine_numbered`. Mục
   [Phần 4](#phần-4--ba-blocker-phải-vá-trước-khi-chạy) có code vá copy-paste được.
3. **Kiến trúc:** notebook 03 chỉ là "vỏ". Cell 2 dùng `%run` để nạp toàn bộ
   `core/shared_pipeline.ipynb` (69 cell: cài package, load model, 3-stage pipeline,
   hàm scoring). Bạn chỉ được viết code từ cell 8 trở xuống.
4. **Nhiệm vụ của bạn:** viết một hàm `stage3_grounding_vNEXT(video_path, t_final)` mới
   tốt hơn `stage3_grounding` gốc, rồi chứng minh `S` mới > `S` baseline trên 20 video
   calibration. **Tuyệt đối không sửa** `core/shared_pipeline.ipynb`.
5. **Thời gian:** lần chạy đầu tiên mất khoảng **50–80 phút** (25–30 phút cho core + 30–50
   phút cho vòng đánh giá 20 video). Mỗi lần thử nghiệm S sau đó chỉ mất **2–5 phút** nếu
   bạn dùng bản vá cache ở [Phần 4.3](#43--patch-b--vòng-lặp-đánh-giá-s-nhanh-hơn-15-lần).

---

## Mục lục

- [Phần 1 — Hệ thống này là gì](#phần-1--hệ-thống-này-là-gì)
- [Phần 2 — Phân tích chi tiết `03_spatial_S.ipynb`](#phần-2--phân-tích-chi-tiết-03_spatial_sipynb)
- [Phần 3 — Chuẩn bị môi trường Kaggle](#phần-3--chuẩn-bị-môi-trường-kaggle)
- [Phần 4 — Ba blocker phải vá trước khi chạy](#phần-4--ba-blocker-phải-vá-trước-khi-chạy)
- [Phần 5 — Quy trình chạy từng bước](#phần-5--quy-trình-chạy-từng-bước)
- [Phần 6 — Sau khi baseline chạy được: làm gì cho metric S](#phần-6--sau-khi-baseline-chạy-được-làm-gì-cho-metric-s)
- [Phần 7 — Xử lý sự cố](#phần-7--xử-lý-sự-cố)
- [Phần 8 — Checklist trước khi mở PR](#phần-8--checklist-trước-khi-mở-pr)

---

## Phần 1 — Hệ thống này là gì

### 1.1 Bài toán

Cuộc thi **ACCIDENT @ CVPR 2026 — AUTOPILOT Workshop** (Kaggle). Cho một clip CCTV cố định
có tai nạn giao thông, dự đoán 3 thứ:

| Ký hiệu | Câu hỏi | Đầu ra | Cách tính điểm |
|---|---|---|---|
| **T** | *When* — tai nạn xảy ra lúc nào? | `accident_time` (giây) | Gaussian, lấy trung bình của 3 mức σ = 0.5 / 1.0 / 2.0 s |
| **S** | *Where* — va chạm ở đâu trong khung? | `center_x`, `center_y` (chuẩn hoá 0–1) | Gaussian **bất đẳng hướng**, σ_x = 0.0952, σ_y = 0.1353 |
| **C** | *What* — loại va chạm gì? | 1 trong 5 nhãn | Top-1 accuracy (0 hoặc 1) |

Điểm cuối cùng là **trung bình điều hoà của 3 giá trị trung bình**:

```
ACCIDENT score = 3 / (1/mean(T) + 1/mean(S) + 1/mean(C))
```

Vì là trung bình điều hoà, **thành phần yếu nhất kéo cả điểm xuống**. Core đã đo được:
cải thiện S từ 0.156 → 0.45 đem lại **+0.111** điểm, gấp ~5 lần so với cải thiện C tương
đương (+0.022). Nói cách khác: **việc bạn đang làm (S) là việc có đòn bẩy lớn nhất trong
cả 3 metric.**

### 1.2 Cấu trúc repo

```text
zero-shot-cctv-accident/
├── README.md                          # quy ước làm việc 3 người, workflow git
├── HUONG_DAN_03_SPATIAL_S.md          # file này
├── core/
│   ├── shared_pipeline.ipynb          # 2.3 MB, 69 cell — TOÀN BỘ hạ tầng dùng chung
│   └── shared_pipeline.py             # bản export .py (2447 dòng) — KHÔNG dùng cho notebook 03
├── notebooks/
│   ├── 01_temporal_T.ipynb            # người phụ trách T
│   ├── 02_type_C.ipynb                # người phụ trách C
│   └── 03_spatial_S.ipynb             # ← BẠN Ở ĐÂY (10 cell)
└── legacy/
    └── classical_pipeline.ipynb       # pipeline cũ (SORT + CLIP + YOLO), giữ làm baseline âm
```

Repo không có `requirements.txt`, không có config YAML, không có script `.sh`/`.ps1`. Mọi
việc cài đặt nằm bên trong `core/shared_pipeline.ipynb` — đây là thiết kế "Kaggle-first".

### 1.3 Pipeline 3 stage (đây là phần bạn phải hiểu rõ)

Với **mỗi video**, `run_inference_vlm` gọi Qwen3-VL-8B-Instruct (4-bit) nhiều lần:

```
video.mp4
   │
   ├─ Stage 1: stage1_full_scan(video, duration, scene)
   │     32 frame @ 2 fps, resize 448px, ĐỐT timestamp "t=xx.xx s" lên góc mỗi frame
   │     → JSON {accident_time, center_x, center_y, type}   ← trả lời cả 3 câu hỏi 1 lượt
   │
   ├─ Stage 2: stage2_time_refine(video, t_base, duration)
   │     cửa sổ dày ±2s @ 4 fps + vài frame context thưa
   │     → t_final = t_base + 0.35 × clip(t_refined − t_base, ±1.5s)   ← hiệu chỉnh CÓ CHẶN
   │
   ├─ Stage 3: stage3_grounding(video, t_final)      ★★★ ĐÂY LÀ PHẦN CỦA BẠN ★★★
   │     ĐÚNG 1 frame tại t_final, resize 768px, KHÔNG đốt timestamp
   │     prompt: '{"point": [x, y]}' với x,y trên thang [0, 1000] (thang native của Qwen3-VL)
   │     → GHI ĐÈ hoàn toàn center_x, center_y của Stage 1 (grounding_weight = 1.0)
   │
   ├─ classify_type_cascade(video, t_final, type_stage1)
   │     3 phiếu (stage1 + prompt 'motion' + prompt 'geometry') + cổng entropy ≤ 1.0
   │     nếu chia 3 phe → thêm 1 lần gọi "elimination" giữa top-2
   │
   └─ apply_scene_type_postfix(type, scene)
         bẻ 't-bone' → 'rear-end' nếu scene ∈ {highway, tunnel, grade_separated_intersection}
```

**Lý do Stage 3 tách riêng khỏi Stage 1** (rất quan trọng cho công việc của bạn): theo
writeup của đội hạng 1 (arXiv:2605.29325), tách grounding thành 1 lần gọi riêng là
**cải tiến đơn lẻ lớn nhất trong toàn bộ quá trình phát triển của họ: +0.09356**. Khi bị hỏi
chung trong Stage 1, model phải chia sự chú ý giữa "khi nào" và "ở đâu", và toạ độ trả về
bị co về một lưới thô (thường là 0.5, 0.5).

### 1.4 Hàm `stage3_grounding` baseline — đối tượng bạn phải đánh bại

Đây là code hiện tại trong core (cell 59). Đọc kỹ vì bạn sẽ viết bản tốt hơn:

```python
def stage3_grounding(video_path, t_final):
    if not VLM_CFG['use_grounding']:
        return None
    # Đúng 1 frame, nên có thể "chi" độ phân giải cao mà Stage 1 không dám
    frames = sample_frames_stamped(video_path, 1.0, t_final, t_final + 1e-3, limit=1,
                                   max_side=VLM_CFG.get('grounding_max_side', 448),
                                   burn=False)          # burn=False: KHÔNG đốt timestamp
    if not frames:
        return None
    text = vlm_generate_oom_safe(GROUNDING_PROMPT, frames, 64, min_frames=1)

    j = _extract_json(text)
    pt = None
    if j:
        if isinstance(j.get('point'), (list, tuple)) and len(j['point']) >= 2:
            pt = (_safe_float(j['point'][0], np.nan), _safe_float(j['point'][1], np.nan))
        elif 'center_x' in j and 'center_y' in j:
            pt = (_safe_float(j['center_x'], np.nan), _safe_float(j['center_y'], np.nan))
    if pt is None:                                      # fallback: regex bắt "(x, y)"
        m = re.search(r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)', text)
        if m:
            pt = (float(m.group(1)), float(m.group(2)))
    if pt is None or any(np.isnan(v) for v in pt):
        return None                                     # None = giữ nguyên toạ độ Stage 1
    x, y = pt
    if x > 1.0 or y > 1.0:                              # thang native [0,1000] → [0,1]
        x, y = x / 1000.0, y / 1000.0
    return (_clip01(x), _clip01(y))
```

Và prompt nó dùng:

```python
GROUNDING_PROMPT = (
    'This CCTV frame shows a traffic accident. '
    'Point to the EXACT location where the collision or impact occurs. '
    'Output the collision point coordinates as JSON:\n'
    '{"point": [x, y]}\n'
    'where x is horizontal (0=left edge, 1000=right edge) and '
    'y is vertical (0=top edge, 1000=bottom edge).'
)
```

### 1.5 Bộ đánh giá: `diverse_videos` (20 video)

Core cell 68 chọn 4 video cho mỗi loại va chạm × 5 loại = **20 video**, từ tập synthetic
CARLA, `random_state=SEED=42` nên **luôn cố định** giữa các lần chạy:

```
head-on   : Town03_head-on_wet_48, Town06_head-on_wet_06, Town03_head-on_night_40, Town06_head-on_wet_01
rear-end  : Town04_rear-end_sunset_13, Town04_rear-end_rain_09, Town05_rear-end_rain_142, Town07_rear-end_rain_28
sideswipe : Town05_sideswipe_clear_04, Town04_sideswipe_wet_10, Town06_sideswipe_wet_06, Town05_sideswipe_night_06
single    : Town10HD_single_sunset_05, Town10HD_single_clear_05, Town07_single_sunset_03, Town10HD_single_clear_12
t-bone    : Town03_t-bone_rain_14, Town10HD_t-bone_sunset_08, Town03_t-bone_rain_23, Town03_t-bone_rain_28
```

Đi kèm là `diverse_labels_df` chứa ground truth. Hai biến này là đầu vào của mọi vòng lặp
đánh giá trong notebook 03.

> **Lưu ý về giới hạn:** n = 20 là rất nhỏ. Với S, sai số chuẩn của trung bình trên 20 mẫu
> khá lớn, nên chênh lệch < 0.03 gần như là nhiễu. Đừng ăn mừng vì `+0.01`.

### 1.6 Mốc "constant floor" — ngưỡng bạn buộc phải vượt

Core cell 50 fit một bộ hằng số tối ưu trên 2211 video synthetic:

```
t = 6.15s   xy = (0.51, 0.51)   type = rear-end
```

Vì σ_S (0.095 / 0.135) xấp xỉ độ tán của chính nhãn (std center_x = 0.13, std center_y = 0.18),
**đoán hằng số (0.51, 0.51) đã là một dự đoán mạnh**. Bất kỳ phiên bản `stage3_grounding`
nào có S thấp hơn constant floor thì nên bị thay bằng hằng số. Đây là bài học đắt nhất
trong repo (mục 7b của core: OWLv2 + optical flow đạt S = 0.156, thua constant 0.22).

---

## Phần 2 — Phân tích chi tiết `03_spatial_S.ipynb`

Notebook có **đúng 10 cell**. Dưới đây là chức năng từng cell, thứ tự chạy, và cell nào sẽ lỗi.

| # | Loại | Nội dung | Trạng thái |
|---|---|---|---|
| 0 | markdown | Giới thiệu mục tiêu: cải thiện S, quy tắc "chỉ định nghĩa hàm MỚI" | OK |
| 1 | code | `[SETUP]` tự tìm `REPO_ROOT`, nếu không có thì `git clone` repo về | OK |
| 2 | code | `%run {CORE_NB_PATH}` — nạp toàn bộ 69 cell của core | OK (chậm, 25–30 phút) |
| 3 | markdown | Giải thích cell 4 | OK |
| 4 | code | `[EVAL]` vẽ histogram phân bố T/S/C của baseline vs constant floor | ❌ **NameError: `eval_core`** |
| 5 | code | `[EVAL]` vẽ đường cong Gaussian scoring (độ nhạy theo σ) | OK — cell duy nhất chạy độc lập |
| 6 | markdown | Gợi ý hướng thí nghiệm: đo content-free bias cho grounding | OK |
| 7 | markdown | Vạch phân cách `🔧 KHU VỰC DEV` | OK |
| 8 | code | Chỉ có comment TODO — chỗ bạn viết `stage3_grounding_vNEXT` | Rỗng |
| 9 | code | `[EVAL]` vòng lặp 20 video, so S mới vs S baseline | ❌ **NameError: `stage2_time_refine_numbered`** |

### 2.1 Cell 1 — cơ chế tìm repo

```python
def _find_or_clone_repo_root():
    cwd = os.getcwd()
    # Nếu đang chạy từ trong repo (notebook ở notebooks/, core/ ở thư mục cha)
    for base in (cwd, os.path.dirname(cwd)):
        if os.path.exists(os.path.join(base, 'core', 'shared_pipeline.ipynb')):
            return os.path.abspath(base)
    # Chưa có → clone về (Kaggle/Colab session mới)
    if not os.path.exists(REPO_NAME):
        subprocess.run(['git', 'clone', '--depth', '1', '-q', REPO_URL], check=True)
    return os.path.abspath(REPO_NAME)
```

Nghĩa là bạn có **2 cách dùng** trên Kaggle:

- **Cách A (khuyến nghị cho người mới):** tạo notebook Kaggle rỗng, copy 10 cell của
  `03_spatial_S.ipynb` vào. Cell 1 sẽ tự `git clone` repo từ GitHub về `/kaggle/working/`.
  Nhược điểm: nó clone branch `main` — **không có code bạn đang sửa dở ở local**.
- **Cách B (khuyến nghị khi bạn đã bắt đầu code):** commit + push branch của bạn lên GitHub,
  rồi sửa `REPO_URL`/thêm `-b feature/spatial-S` vào lệnh clone. Hoặc upload cả repo lên
  Kaggle như một Dataset và mount vào.

### 2.2 Cell 2 — `%run` và điều gì thực sự xảy ra

```python
%run {CORE_NB_PATH}
```

`%run` trên file `.ipynb` là tính năng chính thức của IPython: nó thực thi **toàn bộ 69 cell
của core theo thứ tự**, rồi đổ mọi biến đã định nghĩa vào namespace của notebook 03. Điều này
có nghĩa:

- Mọi biến/hàm của core (`stage3_grounding`, `diverse_videos`, `score_predictions`, `CONST`,
  `PALETTE`, `VLM_CFG`, ...) trở thành biến toàn cục trong notebook của bạn.
- Bạn **không cần** thêm cell `pip install` nào — core cell 8 đã làm.
- Nhưng bạn **không thể bỏ qua** cell nào của core. Bao gồm cả 2 cell "sửa môi trường"
  có tác dụng phụ nguy hiểm (xem [Phần 4.4](#44--patch-c-tuỳ-chọn--vô-hiệu-hoá-cell-hạ-cấp-torch-của-core)).
- Nếu core lỗi giữa đường, trạng thái nạp được có thể **không đầy đủ**. Đừng chữa cháy bằng
  cách chạy tiếp các cell dưới — hãy sửa nguyên nhân rồi chạy lại cell 2 **từ đầu**.

> **Lưu ý về mức độ đã được kiểm chứng:** cả 3 notebook `01`/`02`/`03` đều **không có output
> nào được lưu ở cell 2**, nghĩa là đường đi `%run core` chưa từng được chạy trọn vẹn và lưu
> lại trong repo (notebook 02 nặng 1.66 MB vì output của các cell thí nghiệm độc lập, không
> phải của cell 2). Đây cũng là lý do 3 blocker ở [Phần 4](#phần-4--ba-blocker-phải-vá-trước-khi-chạy)
> tồn tại mà chưa ai phát hiện. Bạn rất có thể là người đầu tiên chạy trọn vẹn đường đi này —
> hãy chạy từng cell một và đối chiếu với bảng kiểm ở [Bước 4](#bước-4--chạy-cell-1-và-cell-2-và-những-gì-phải-kiểm-tra-dọc-đường).

**Thứ tự các mốc quan trọng khi `%run` chạy:**

| Cell core | Việc | Thời gian |
|---|---|---|
| 6 | Tự dò `BASE_DIR` (`/kaggle/input/competitions/accident` hoặc `/kaggle/input/accident`) | tức thì |
| 7 | Audit đường dẫn — **bảng này phải hiện `exists=True` toàn bộ** | tức thì |
| 8 | Smoke-test torch trong subprocess + `pip install` ultralytics/CLIP/perception_models/transformers | 3–6 phút |
| 10–39 | Load `labels.csv` (2211 dòng), clean, EDA, vẽ biểu đồ | 1–3 phút |
| 42–43 | Load YOLOv8s (22 MB) + CLIP ViT-B/32 (338 MB) | 1–2 phút |
| 52 | Tải + load **Qwen3-VL-8B-Instruct 4-bit** (~16 GB tải về, ~6.9 GB VRAM) | 8–15 phút |
| 55 | `[FIX]` cố cài `nvidia-cuda-nvrtc-cu13` — **sẽ FAIL, vô hại** | 1–2 phút |
| 57 | `[FIX v2]` **force-reinstall torch về cu121** — có tác dụng phụ | 3–5 phút |
| 58 | `[CANARY]` kiểm tra VLM thực sự sinh được text (`'OK'`) | 5 giây |
| 59, 61, 62, 63 | Định nghĩa 3 stage + cascade + `run_inference_vlm` | tức thì |
| 67 | Định nghĩa `temporal_score` / `spatial_score` / `accident_score` / `score_predictions` | tức thì |
| 68 | Dựng `diverse_videos` + `diverse_labels_df` (20 video) | tức thì |

**Core kết thúc ở cell 68.** Đây chính là gốc rễ của blocker số 1: core **không hề có** cell
nào chạy pipeline trên 20 video để tạo ra `eval_core`.

### 2.3 Cell 4 — cái gì thiếu

```python
if not eval_core.empty:                                    # ← NameError ngay tại đây
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, name, color in zip(axes, score_cols, score_names, colors):
        sns.histplot(eval_core[col], bins=20, kde=True, color=color, ax=ax)
        ax.axvline(eval_core[col].mean(), linestyle='--', color='black')
        ax.axvline(const_eval[col].mean(), linestyle=':', color='crimson')   # ← và ở đây
```

Cell 4 cần 2 DataFrame:

- `eval_core` — kết quả pipeline baseline trên 20 video, đã tính T/S/C từng video.
- `const_eval` — kết quả của bộ hằng số `CONST` trên cùng 20 video (constant floor).

Cả hai **không được định nghĩa ở bất kỳ đâu trong repo**. Notebook 01 cũng có cùng lỗi này
(markdown của nó ghi *"T ghi trong `eval_core['T']` ở cell cuối của core"* — nhưng cell cuối
của core là cell 68, không có `eval_core`). Kết luận: người viết core đã **quên export cell
đánh giá baseline** khi modular hoá repo. Bạn phải tự dựng lại — [Patch A](#42--patch-a--dựng-eval_core-và-const_eval).

### 2.4 Cell 9 — vòng lặp đánh giá, và 3 vấn đề của nó

```python
rows_S_new = []
for vp, gt_type in zip(diverse_videos, diverse_labels_df['type']):
    sub_path = 'videos/' + vp.name
    scene = SCENE_BY_PATH.get(sub_path)
    cap = cv2.VideoCapture(str(vp))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = (n / fps) if fps > 0 else 20.0

    pred = stage1_full_scan(vp, duration, scene)                               # giữ nguyên (core)
    t_final = stage2_time_refine_numbered(vp, pred['accident_time'], duration)  # ← NameError

    # <-- đổi dòng dưới đây sang hàm mới của bạn -->
    pt = stage3_grounding(vp, t_final)
    if pt is not None:
        pred['center_x'], pred['center_y'] = pt

    pred['type'] = classify_type_cascade(vp, t_final, pred['type'])
    pred['type'] = apply_scene_type_postfix(pred['type'], scene)

    rows_S_new.append({...})

eval_S_new = score_predictions(pd.DataFrame(rows_S_new), diverse_labels_df)
print(f"[BASELINE] S = {eval_core['S'].mean():.4f}")
print(f"[MOI]      S = {eval_S_new['S'].mean():.4f}")
```

Ba vấn đề, theo thứ tự nghiêm trọng:

1. **`stage2_time_refine_numbered` không tồn tại.** Core chỉ có `stage2_time_refine`. Tên
   `_numbered` là dấu tích của thí nghiệm NumPro trong notebook 01 (đốt *số thứ tự frame*
   thay vì *số giây*) chưa bao giờ được đưa vào core. → Patch A sẽ tạo alias.
2. **Cực kỳ tốn GPU một cách vô ích.** Vòng lặp này chạy lại Stage 1 + Stage 2 + cascade
   loại va chạm cho cả 20 video, **mỗi lần bạn thử một ý tưởng Stage 3 mới**. Nhưng tất cả
   các stage đó đều dùng greedy decode (`do_sample=False`) nên **kết quả lặp lại y hệt**.
   Bạn đang trả 30–50 phút GPU để tính lại thứ không đổi. → Patch B giải quyết.
3. **`SCENE_BY_PATH.get('videos/' + vp.name)` luôn trả `None` trên tập calibration.**
   `SCENE_BY_PATH` được dựng từ `test_metadata.csv` (2027 video CCTV **thật**), còn
   `diverse_videos` là video **synthetic** ở `videos/head-on/Town03_....mp4`. Tên không khớp
   → `scene = None` → Stage 1 không có scene hint, và `apply_scene_type_postfix` không bao
   giờ kích hoạt. Đây không phải bug cần sửa (tập synthetic không có `scene_layout`), nhưng
   bạn cần biết: **điểm đo trên calibration set là điểm KHÔNG có scene hint.**

Ngoài ra `gt_type` trong `zip(...)` không được dùng ở đâu — vô hại.

### 2.5 Cell 6 — hướng thí nghiệm mà notebook đã gợi ý

Đây là gợi ý quan trọng nhất trong notebook, dịch lại cho rõ:

> Dựa trên bài học đã rút ra ở `02_type_C` (VLM bị *collapse* khi phải chọn từ một tập giá
> trị neo rời rạc, **kể cả khi neo là số**): `stage3_grounding` hiện tại hỏi thẳng
> `center_x, center_y` — rủi ro tương tự có thể xảy ra (model đoán về (0.5, 0.5) hoặc vài
> toạ độ neo quen thuộc bất kể nội dung). Thí nghiệm content-free đầu tiên nên làm ở đây:
> đo `stage3_grounding` trên khung hình **PRE-COLLISION thật** — nếu toạ độ trả về vẫn hội
> tụ mạnh về 1–2 điểm cố định ngay cả khi chưa có va chạm, đó là cùng một loại bias đã thấy
> ở C, không phải lỗi grounding thật.

Bối cảnh: notebook `02_type_C` đã áp dụng phương pháp *Calibrate Before Use* (Zhao et al.,
ICML 2021) và đo được prior content-free của model rất lệch:

```
t-bone: 0.48   sideswipe: 0.24   single: 0.16   rear-end: 0.08   head-on: 0.04
```

Tức là model thiên vị `t-bone` gấp 12 lần `head-on` **ngay cả khi không có thông tin hình
ảnh nào**. Câu hỏi mở cho bạn: liệu grounding có bias tương tự về tâm khung hình không?

---

## Phần 3 — Chuẩn bị môi trường Kaggle

### 3.1 Tại sao không chạy được ở máy local

Máy bạn: Windows 10/11, Python 3.9.7, không có dataset. Ba lý do chặn cứng:

1. **Đường dẫn dataset hardcode cho Kaggle.** Core cell 6 chỉ dò
   `/kaggle/input/competitions/accident` và `/kaggle/input/accident`. Không tìm thấy → in
   `[ERROR] Could not auto-detect the dataset root` → `labels_df` rỗng → cell 21 trở đi
   sụp theo chuỗi.
2. **VRAM.** Qwen3-VL-8B ở 4-bit cần ~6.9 GB VRAM chỉ để chứa weight, cộng activation cho
   32 frame × 448 px (≈ 4.600 visual token). Nếu không có GPU, core cell 52 sẽ
   `raise RuntimeError('Qwen3-VL is not usable...')` và dừng hẳn.
3. **`OUTPUT_DIR = pathlib.Path('/kaggle/working')`.** Trên Windows lệnh này tạo ra
   `C:\kaggle\working` — không lỗi ngay nhưng là dấu hiệu rõ ràng code không được viết cho
   local.

> Nếu bạn *thật sự* muốn chạy local với GPU ≥ 16 GB: phải tải dataset về, tạo cấu trúc
> `<root>/sim_dataset/{labels.csv, videos/<type>/*.mp4}` + `<root>/{test_metadata.csv,
> sample_submission.csv}`, rồi sửa `_CANDIDATE_ROOTS` trong core cell 6. Đây là sửa core →
> theo README phải mở branch `core/<mô-tả>` riêng. Không khuyến nghị.

### 3.2 Cấu hình Kaggle notebook

Vào [kaggle.com](https://www.kaggle.com) → **Create → Notebook**, rồi ở panel bên phải:

| Mục | Giá trị | Ghi chú |
|---|---|---|
| **Accelerator** | `GPU T4 x2` | Core cell 52 tự chọn GPU trống nhất. T4 x2 cho ~15 GB trống trên GPU còn lại — an toàn nhất. `P100` cũng chạy được. |
| **Internet** | `On` | **Bắt buộc.** Cần cho `pip install`, `git clone`, tải model từ HuggingFace. Yêu cầu tài khoản đã xác thực số điện thoại. |
| **Persistence** | `Files only` (nên bật) | Giúp cache `rows_core.csv` sống qua nhiều session. |
| **Add Input → Competition** | `accident` (ACCIDENT @ CVPR 2026) | Phải **Join Competition** + đồng ý rules trước. |
| **Environment** | Latest / mặc định | Core tự sửa torch nếu cần. |

Sau khi add input, kiểm tra ở panel Data bên phải phải thấy đường dẫn dạng
`/kaggle/input/competitions/accident/` với các mục `sim_dataset/`, `videos/`,
`test_metadata.csv`, `sample_submission.csv`.

### 3.3 (Nên làm) Thêm `HF_TOKEN` để tải model nhanh hơn

Log của core cho thấy:

```
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

Cách khắc phục: tạo token đọc ở [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
rồi trên Kaggle vào **Add-ons → Secrets**, tạo secret tên `HF_TOKEN`. Thêm cell này
**trước cell 1** của notebook:

```python
# [SETUP] HF token -- giup tai Qwen3-VL-8B nhanh hon va tranh rate limit
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
    print('[SETUP] HF_TOKEN da nap')
except Exception as e:
    print(f'[SETUP] Khong nap duoc HF_TOKEN ({e}) -- van chay duoc, chi cham hon')
```

### 3.4 Kiểm tra hạn mức GPU

Kaggle cho **30 giờ GPU/tuần**, mỗi session tối đa 12 giờ. Lần chạy đầy đủ đầu tiên tốn
~1–1,5 giờ. Với cache của Patch A, mỗi thí nghiệm sau đó tốn ~30 phút (chủ yếu là nạp lại
core) + 2–5 phút cho vòng đánh giá S. Bạn có thừa quota, nhưng **đừng để session GPU chạy
không** — bấm *Stop Session* khi nghỉ.

---

## Phần 4 — Ba blocker phải vá trước khi chạy

### 4.1 Tổng hợp blocker

Tôi đã đối chiếu từng tên biến/hàm mà notebook 03 sử dụng với 69 cell của core:

| Tên | Notebook 03 dùng ở | Core định nghĩa? | Hậu quả |
|---|---|---|---|
| `eval_core` | cell 4, cell 9 | ❌ Không | `NameError` — không có điểm baseline để so |
| `const_eval` | cell 4 | ❌ Không | `NameError` — không vẽ được constant floor |
| `stage2_time_refine_numbered` | cell 9 | ❌ Không (chỉ có `stage2_time_refine`) | `NameError` giữa vòng lặp |
| `predict_accident_time_ensemble` | — (core cell 63 gọi) | ❌ Không | **`run_inference_vlm` của core bị hỏng** — đừng gọi nó |
| `stage1_full_scan` | cell 9 | ✅ cell 59 | OK |
| `stage3_grounding` | cell 9 | ✅ cell 59 | OK |
| `classify_type_cascade` | cell 9 | ✅ cell 62 | OK |
| `apply_scene_type_postfix` | cell 9 | ✅ cell 59 | OK |
| `score_predictions`, `accident_score` | cell 4, 9 | ✅ cell 67 | OK |
| `diverse_videos`, `diverse_labels_df` | cell 9 | ✅ cell 68 | OK |
| `SCENE_BY_PATH` | cell 9 | ✅ cell 53 | OK (nhưng luôn trả `None` — xem 2.4) |
| `PALETTE` | cell 4 | ✅ cell 5 | OK |

**Cảnh báo riêng về `run_inference_vlm`:** core cell 63 gọi `predict_accident_time_ensemble`
(bước "classical anchor") nhưng hàm này không tồn tại ở đâu trong repo — kể cả trong
`legacy/classical_pipeline.ipynb`. Nghĩa là `run_inference_vlm` **chắc chắn crash** nếu bạn
gọi. May mắn là notebook 03 không gọi nó; cell 9 gọi từng stage trực tiếp. **Đừng "tiện tay"
thay vòng lặp cell 9 bằng `run_inference_vlm`.** Nếu cần, hãy báo cho người tổng hợp core.

---

### 4.2 Patch A — dựng `eval_core` và `const_eval`

**Chèn cell này ngay sau cell 2 (`%run`), trước cell 3/4.**

Nó làm 3 việc: (a) tạo alias `stage2_time_refine_numbered`, (b) chạy pipeline baseline trên
20 video và **cache ra CSV**, (c) tính `eval_core` + `const_eval`.

```python
# [PATCH A] Dung eval_core + const_eval -- core notebook KHONG co cell nay.
# Lan dau chay het 30-50 phut GPU; ket qua duoc cache ra /kaggle/working nen
# nhung lan sau chi doc lai file, mat 1 giay.
import pathlib, time
import pandas as pd
import cv2

CACHE_CORE = pathlib.Path('/kaggle/working/rows_core_baseline.csv')


def stage2_time_refine_numbered(video_path, t_base, duration):
    """Alias cho stage2_time_refine.

    Notebook 01 va 03 deu goi ten nay, nhung core chi dinh nghia
    stage2_time_refine. Ten '_numbered' la dau tich cua thi nghiem NumPro
    (dot so thu tu frame thay vi so giay) chua bao gio duoc merge vao core.
    Giu nguyen hanh vi baseline de diem S do duoc la diem so sanh dung.
    """
    return stage2_time_refine(video_path, t_base, duration)


def _video_duration(vp):
    cap = cv2.VideoCapture(str(vp))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (n / fps) if fps > 0 else 20.0


if CACHE_CORE.exists():
    rows_core = pd.read_csv(CACHE_CORE).to_dict('records')
    print(f'[CACHE] Doc lai {len(rows_core)} dong baseline tu {CACHE_CORE}')
else:
    rows_core, _t0 = [], time.time()
    for _i, vp in enumerate(diverse_videos, 1):
        scene    = SCENE_BY_PATH.get('videos/' + vp.name)   # None tren tap synthetic
        duration = _video_duration(vp)

        pred    = stage1_full_scan(vp, duration, scene)
        t_final = stage2_time_refine_numbered(vp, pred['accident_time'], duration)
        pt      = stage3_grounding(vp, t_final)
        if pt is not None:
            pred['center_x'], pred['center_y'] = pt
        pred['type'] = classify_type_cascade(vp, t_final, pred['type'])
        pred['type'] = apply_scene_type_postfix(pred['type'], scene)

        rows_core.append({'path': str(vp), 'accident_time': t_final,
                          'center_x': pred['center_x'], 'center_y': pred['center_y'],
                          'type': pred['type'], 'duration': duration})
        print(f'[{_i:2d}/{len(diverse_videos)}] {vp.name:42s} '
              f"t={t_final:6.2f}s  xy=({pred['center_x']:.3f},{pred['center_y']:.3f})  "
              f"{pred['type']:10s} ({time.time()-_t0:5.0f}s)")

    pd.DataFrame(rows_core).to_csv(CACHE_CORE, index=False)
    print(f'[CACHE] Da luu baseline vao {CACHE_CORE}')

# Diem baseline cua core
eval_core = score_predictions(pd.DataFrame(rows_core), diverse_labels_df)

# Constant floor: CONST da duoc fit o core cell 50 tren 2211 video synthetic
const_eval = score_predictions(
    pd.DataFrame([{'path': str(vp), **CONST} for vp in diverse_videos]),
    diverse_labels_df)

print(f"\n[BASELINE core ] T={eval_core['T'].mean():.4f}  S={eval_core['S'].mean():.4f}  "
      f"C={eval_core['C'].mean():.4f}  ACCS={accident_score(eval_core):.4f}")
print(f"[CONSTANT floor] T={const_eval['T'].mean():.4f}  S={const_eval['S'].mean():.4f}  "
      f"C={const_eval['C'].mean():.4f}  ACCS={accident_score(const_eval):.4f}")
print(f"\n[S CAN VUOT] baseline={eval_core['S'].mean():.4f}  "
      f"floor={const_eval['S'].mean():.4f}  -> muc tieu > "
      f"{max(eval_core['S'].mean(), const_eval['S'].mean()):.4f}")
```

**Ghi lại 2 con số** `S baseline` và `S floor` in ra ở cuối — đó là hai mốc bạn phải vượt.
Ghi ngay vào markdown cell đầu notebook theo đúng format README yêu cầu:

```markdown
## Kết quả (cập nhật lần cuối: <ngày>)
Baseline S (core) : 0.xxxx
Constant floor S  : 0.xxxx
Sau thử nghiệm    : 0.xxxx  (+/- x.xxxx)
```

> **Về cache:** nếu bạn bật Persistence = `Files only`, file `rows_core_baseline.csv` sống
> qua các session ⇒ chỉ phải trả 30–50 phút GPU **một lần duy nhất**. Nếu bạn sửa bất kỳ
> thứ gì ảnh hưởng Stage 1/2 (hoặc core được update), hãy xoá file cache để tính lại.

---

### 4.3 Patch B — vòng lặp đánh giá S nhanh hơn 15 lần

**Thay toàn bộ nội dung cell 9 bằng code dưới đây.**

Ý tưởng: để đo S, bạn **chỉ cần chạy lại Stage 3**. `t_final` và `type` đã có trong
`rows_core` và sẽ không đổi (mọi lần gọi VLM đều `do_sample=False`, tức greedy, nên tất định).
Vòng lặp gốc chạy lại Stage 1 + Stage 2 + 3–4 lần gọi cascade cho mỗi video — khoảng
**95% thời gian GPU bị đốt vào việc tính lại thứ không thay đổi**.

```python
# [PATCH B] Danh gia S -- CHI chay lai Stage 3, tai su dung t_final va type tu
# rows_core (Patch A). Moi lan thu 1 y tuong grounding chi mat ~2-5 phut thay vi
# 30-50 phut: Stage 1/2/cascade dung greedy decode nen ket qua tat dinh, chay lai
# chung khong doi gi ngoai tien GPU.
#
# CHU Y: doi ten ham o dong duoi thanh ham cua ban.
GROUNDING_FN = stage3_grounding          # <-- doi thanh stage3_grounding_vNEXT

rows_S_new, n_none = [], 0
for r in rows_core:
    vp      = pathlib.Path(r['path'])
    t_final = float(r['accident_time'])

    pt = GROUNDING_FN(vp, t_final)
    if pt is None:                       # None = giu nguyen toa do cua Stage 1
        n_none += 1
        cx, cy = float(r['center_x']), float(r['center_y'])
    else:
        cx, cy = pt

    rows_S_new.append({'path': r['path'], 'accident_time': t_final,
                       'center_x': cx, 'center_y': cy, 'type': r['type']})

eval_S_new = score_predictions(pd.DataFrame(rows_S_new), diverse_labels_df)

_s_old, _s_new = eval_core['S'].mean(), eval_S_new['S'].mean()
print(f'[BASELINE] S = {_s_old:.4f}')
print(f'[MOI]      S = {_s_new:.4f}')
print(f'[DELTA]        {_s_new - _s_old:+.4f}')
print(f'[FLOOR]    S = {const_eval["S"].mean():.4f}  (constant 0.51/0.51)')
print(f'[PARSE]    {n_none}/{len(rows_S_new)} video grounding tra None (fallback Stage 1)')
print(f'\n[ACCS] baseline={accident_score(eval_core):.4f} -> moi={accident_score(eval_S_new):.4f}')

# Bang chi tiet tung video -- de biet cai thien den tu dau, hay chi 1-2 video may man
_cmp = eval_core[['video_stem', 'type_gt', 'S']].rename(columns={'S': 'S_base'}).merge(
    eval_S_new[['video_stem', 'S']].rename(columns={'S': 'S_new'}), on='video_stem')
_cmp['delta'] = _cmp['S_new'] - _cmp['S_base']
display(_cmp.sort_values('delta').round(4))
```

Bảng cuối rất quan trọng: nó cho biết cải thiện của bạn là **hệ thống** (đa số video tốt lên)
hay chỉ là **may mắn trên 1–2 video**. Với n = 20, một video từ 0.02 → 0.95 đã đủ đẩy trung
bình lên +0.046 mà không chứng minh được gì.

---

### 4.4 Patch C (tuỳ chọn) — vô hiệu hoá cell hạ cấp torch của core

**Vấn đề.** Core cell 55 và 57 là hai cell "sửa môi trường" viết cho một sự cố cụ thể trong
quá khứ (torch cu130 thiếu `libnvrtc-builtins.so.13.0`). Nhưng chúng **chạy vô điều kiện**
mỗi lần `%run`:

- **Cell 55** cố `pip install nvidia-cuda-nvrtc-cu13` → thất bại ở bước build wheel
  (`exit code 1`). Vô hại, chỉ mất 1–2 phút.
- **Cell 57** thì khác: nó `pip install --force-reinstall torch torchvision torchaudio
  --index-url .../cu121`, **hạ torch từ 2.10+cu128 xuống 2.5.1+cu121** trên đĩa, rồi in
  `[ACTION REQUIRED] Restart Kernel now`.

Tại sao đây là mìn: trong session hiện tại, torch đã được `import` vào bộ nhớ **trước** cell
57, nên mọi thứ vẫn chạy (log của core xác nhận canary PASS). Nhưng:

- Mất 3–5 phút và ~2,5 GB băng thông mỗi lần chạy, không đổi lại gì.
- Nếu kernel restart vì bất kỳ lý do gì (OOM, bạn bấm Restart, session hết hạn), môi trường
  sẽ là **torch 2.5.1 + transformers 5.14** — gần như chắc chắn không tương thích, và
  `bitsandbytes` sẽ lỗi khi nạp 4-bit.

**Cách xử lý — chọn 1 trong 2:**

**Cách 1 (không sửa core, an toàn nhất về mặt git):** cứ để nó chạy, nhưng
**TUYỆT ĐỐI KHÔNG bấm Restart Kernel** giữa phiên. Nếu buộc phải restart, hãy chạy lại từ
cell 1 và chấp nhận rằng cell 8 sẽ cài lại torch từ image gốc.

**Cách 2 (khuyến nghị nếu bạn chạy nhiều lần):** thêm cổng điều kiện cho core cell 57. Theo
README **Bước 4**, sửa core phải làm trên branch riêng (ví dụ `core/gate-torch-downgrade`) và
mở PR riêng có cả 3 người review. Thay **toàn bộ** nội dung cell 57 bằng:

```python
# [FIX v2] Ha torch ve cu121 -- CHI khi smoke-test o cell 8 that su bao loi.
#
# Truoc day cell nay chay vo dieu kien: moi lan %run no ha torch 2.10+cu128 ->
# 2.5.1+cu121 tren dia, mat 3-5 phut va ~2.5 GB bang thong ma khong doi lai gi
# (session hien tai da import torch cu vao bo nho roi). Dat min o cho: neu kernel
# restart sau do, moi truong con lai la torch 2.5.1 + transformers 5.x, va
# bitsandbytes se hong khi nap 4-bit. Cong duoi day dung chinh ket qua smoke test
# cua cell 8, nen moi truong hong thi van duoc sua y nhu cu.
import subprocess, sys

if _after.get('cuda_op_ok') is True:
    print('[SKIP] Smoke test o cell 8 bao cuda_op_ok=True -- torch hien tai lanh, '
          'bo qua buoc ha ve cu121.')
else:
    print('[STATUS] Force-reinstalling torch/torchvision/torchaudio tu cu121 index...')
    _rc = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
         'torch', 'torchvision', 'torchaudio',
         '--index-url', 'https://download.pytorch.org/whl/cu121'],
        capture_output=True, text=True)
    print(f'[STATUS] cu121 install exit code: {_rc.returncode}')
    if _rc.returncode != 0:
        print(_rc.stderr[-2000:])
    else:
        # Re-pin constraints file de cac install sau khong keo cu130 tro lai
        _pinned = []
        for _pkg in ('torch', 'torchvision', 'torchaudio'):
            _show = subprocess.run([sys.executable, '-m', 'pip', 'show', _pkg],
                                   capture_output=True, text=True)
            for _line in _show.stdout.splitlines():
                if _line.startswith('Version:'):
                    _pinned.append(f"{_pkg}=={_line.split(':', 1)[1].strip()}")
        with open('/kaggle/working/_torch_constraints.txt', 'w') as f:
            f.write('\n'.join(_pinned) + '\n')
        print(f'[STATUS] Re-pinned constraints for downstream installs: {_pinned}')
    print('\n[ACTION REQUIRED] Restart Kernel now, then run from the top.')
```

`_after` là biến do chính core cell 8 tạo ra (kết quả smoke test *sau khi* cài package), nên
cổng này tự thích ứng: môi trường lành thì bỏ qua, môi trường hỏng thì vẫn sửa như cũ.

---

## Phần 5 — Quy trình chạy từng bước

### Bước 0 — Tạo branch của bạn (làm ở local, trước khi lên Kaggle)

Repo hiện đang ở branch `main` và 3 notebook đang có thay đổi chưa commit. Xem trước rồi
tạo branch riêng theo đúng quy ước README:

```powershell
cd C:\Users\Khuong\Desktop\ThuHuyen\zero-shot-cctv-accident
git status
git diff --stat
git checkout -b feature/spatial-S
```

### Bước 1 — Đưa notebook lên Kaggle

Chọn 1 trong 2 cách:

**Cách A — Upload trực tiếp (nhanh nhất để bắt đầu):**
Kaggle → **Create → Notebook** → menu **File → Import Notebook** → upload
`notebooks/03_spatial_S.ipynb`. Cell 1 sẽ tự `git clone` core từ GitHub branch `main`.

**Cách B — Push branch rồi clone (khi bạn đã có code riêng):**

```powershell
git push -u origin feature/spatial-S
```

Rồi trên Kaggle sửa cell 1, thay lệnh clone thành:

```python
subprocess.run(['git', 'clone', '--depth', '1', '-q',
                '-b', 'feature/spatial-S', REPO_URL], check=True)
```

### Bước 2 — Cấu hình session

Theo bảng ở [Phần 3.2](#32--cấu-hình-kaggle-notebook): GPU T4 x2, Internet On, add
competition input `accident`, Persistence `Files only`. (Tuỳ chọn: thêm cell `HF_TOKEN` ở
[Phần 3.3](#33-nên-làm--thêm-hf_token-để-tải-model-nhanh-hơn).)

### Bước 3 — Chèn các patch vào notebook

Thứ tự cell sau khi vá:

```
cell 0   markdown  giới thiệu
cell 0.5 code      (tuỳ chọn) nạp HF_TOKEN
cell 1   code      [SETUP] tìm/clone repo
cell 2   code      %run core notebook                      ← 25-30 phút
cell 2.5 code      ★ PATCH A (dựng eval_core/const_eval)   ← 30-50 phút lần đầu
cell 3   markdown
cell 4   code      [EVAL] histogram T/S/C — giờ đã chạy được
cell 5   code      [EVAL] đường cong Gaussian
cell 6   markdown  gợi ý hướng thí nghiệm
cell 7   markdown  🔧 KHU VỰC DEV
cell 8   code      ★ code của bạn: stage3_grounding_vNEXT
cell 9   code      ★ PATCH B (vòng lặp đánh giá S nhanh)
```

### Bước 4 — Chạy cell 1 và cell 2, và những gì phải kiểm tra dọc đường

Chạy **từng cell một** (Shift+Enter), đừng bấm *Run All* ở lần đầu. Trong lúc cell 2 chạy,
đây là các mốc phải xác nhận — nếu một mốc sai, **dừng lại ngay**, đừng để nó chạy tiếp:

| Mốc | Log phải thấy | Nếu sai |
|---|---|---|
| Path detect | `BASE_DIR : /kaggle/input/competitions/accident` | Chưa add competition input, hoặc chưa join competition |
| Path audit | Bảng có **toàn bộ** `exists=True` (màu xanh) | Dataset mount sai layout |
| Smoke test | `'cuda_op_ok': True` | Xem [Phần 7](#phần-7--xử-lý-sự-cố) |
| Cài package | 5 dòng `install exit code: 0` | Internet chưa bật |
| transformers | `qwen3_vl architecture recognized: True` | transformers < 4.57 → Restart Kernel rồi chạy lại |
| Load labels | `synthetic_labels  2211  19` và `test_metadata  2027  10` | `labels.csv` không đọc được |
| YOLO | `[SUCCESS] YOLOv8s loaded` | Internet |
| CLIP | `[SUCCESS] CLIP ViT-B/32 loaded \| text features for 5 types` | Internet |
| Qwen3-VL | `[SUCCESS] Qwen/Qwen3-VL-8B-Instruct loaded` | Xem [Phần 7](#phần-7--xử-lý-sự-cố) |
| Quantization | `Linear4bit layers: 368` (**phải khác 0**) | Nếu = 0 → model đang ở fp16, sẽ OOM về sau |
| VRAM | `GPU1 ... 6.95 GB used / 15.64 GB total, 8.69 GB free` | < 4 GB free → sẽ OOM |
| Canary | `[CANARY] generation returned: 'OK'` + `PASS -- the model is alive` | **Không đi tiếp nếu fail** |
| 3 stage | `[STATUS] stage1_full_scan / stage2_time_refine / stage3_grounding defined` | — |
| Scoring | `[METRIC] sigma_t = (0.5, 1.0, 2.0) \| sigma_x = 0.0952 sigma_y = 0.1353` | — |
| Calibration set | `[STATUS] Diverse calibration set: 20 videos across 5 types` | — |

> **Canary là chốt an toàn quan trọng nhất.** Nó tồn tại vì đã từng có lần model "load thành
> công" nhưng mọi lần gọi trả về chuỗi rỗng, và pipeline lặng lẽ ghi giá trị mặc định
> (`0.35×duration`, `0.5/0.5`, `rear-end`) cho toàn bộ 2027 clip — một hằng số đội lốt
> dự đoán, sau 24 phút GPU. Nếu canary fail, mọi con số phía sau đều vô nghĩa.

### Bước 5 — Chạy Patch A, ghi lại baseline

Cell này in tiến độ từng video. Xác nhận 2 điều:

1. **Không có `[WARNING] OOM`** liên tục. Một hai lần thì chấp nhận được (retry sẽ giảm số
   frame), nhưng nếu mọi video đều OOM thì Stage 1 đang trả lời từ 6–12 frame trải trên 20 s —
   điểm đo ra sẽ vô nghĩa. Xem [Phần 7](#phần-7--xử-lý-sự-cố).
2. **Toạ độ `xy=` phải đa dạng.** Nếu cả 20 video đều ra `xy=(0.500,0.500)` thì Stage 1 hoặc
   Stage 3 đang không parse được JSON — đó chính là bias content-free mà cell 6 cảnh báo, và
   cũng là phát hiện đầu tiên đáng giá cho nhiệm vụ của bạn.

Ghi lại `S baseline` và `S floor`.

### Bước 6 — Chạy cell 4 và 5, đọc biểu đồ

- **Cell 4** cho 3 histogram T / S / C, mỗi cái có 2 đường dọc: `pipeline` (nét đứt đen) và
  `constant` (nét chấm đỏ). Với S, hãy xem **hình dạng phân bố**: nếu S phân cực về 0 (nhiều
  video gần 0, vài video gần 1) thì lỗi mang tính "sai hẳn khu vực"; nếu S dồn ở giữa
  (0.3–0.6) thì lỗi là lệch nhỏ có hệ thống — hai chẩn đoán này dẫn tới hai hướng sửa hoàn
  toàn khác nhau.
- **Cell 5** cho đường cong độ nhạy. Con số cần nhớ: với σ_x = 0.095, sai 0.1 theo trục x
  (≈10% chiều rộng khung) đã làm S rơi xuống ~0.57; sai 0.2 thì còn ~0.11. **Grounding phải
  chính xác trong khoảng ~10% chiều rộng khung mới có điểm tốt.**

### Bước 7 — Viết hàm mới ở cell 8, đánh giá ở cell 9

Xem [Phần 6](#phần-6--sau-khi-baseline-chạy-được-làm-gì-cho-metric-s) để biết nên thử gì.
Quy tắc bất di bất dịch:

- Đặt tên hàm **mới** (`stage3_grounding_v2`, `stage3_grounding_v3`, ...). Không gán lại
  `stage3_grounding = ...`, vì `eval_core` được tính bằng bản gốc và bạn sẽ mất mốc so sánh.
- Chỉ đổi biến `GROUNDING_FN` ở đầu Patch B.
- Chạy lại **chỉ cell 8 và cell 9** cho mỗi ý tưởng — không cần chạy lại cell 2 hay Patch A.

---

## Phần 6 — Sau khi baseline chạy được: làm gì cho metric S

Notebook 03 là notebook "trắng" nhất trong 3 — chưa có lịch sử thí nghiệm nào cho
`stage3_grounding`. Dưới đây là các hướng theo thứ tự ưu tiên tôi khuyến nghị, dựa trên
những gì repo đã chứng minh và chưa chứng minh.

### 6.1 Việc đầu tiên: đo content-free bias (chẩn đoán, không phải cải tiến)

Đây chính là gợi ý ở cell 6, và nên làm trước mọi thứ khác — vì nó quyết định bạn đang sửa
loại lỗi nào. Ý tưởng: gọi `stage3_grounding` trên frame **trước khi va chạm** (`t_final − 3s`
chẳng hạn, khi chưa có gì xảy ra). Nếu toạ độ trả về vẫn hội tụ về vài điểm cố định, thì lỗi
là **bias của prior**, không phải model "nhìn không ra".

```python
# [DIAG] Do content-free bias cua grounding -- KHONG phai cai tien, la chan doan.
# Neu toa do tren frame PRE-COLLISION giong het toa do tren frame va cham, model
# dang tra ve prior cua no chu khong phai doc anh.
from collections import Counter

diag = []
for r in rows_core:
    vp, t_final = pathlib.Path(r['path']), float(r['accident_time'])
    t_pre = max(0.0, t_final - 3.0)          # 3 giay TRUOC va cham

    pt_impact = stage3_grounding(vp, t_final)
    pt_pre    = stage3_grounding(vp, t_pre)
    diag.append({'video': vp.stem,
                 'x_impact': None if pt_impact is None else round(pt_impact[0], 3),
                 'y_impact': None if pt_impact is None else round(pt_impact[1], 3),
                 'x_pre':    None if pt_pre    is None else round(pt_pre[0], 3),
                 'y_pre':    None if pt_pre    is None else round(pt_pre[1], 3)})

diag_df = pd.DataFrame(diag)
display(diag_df)

# Toa do tra ve co bao nhieu gia tri PHAN BIET? Neu <= 3-4 tren 20 video thi
# model dang luong tren mot luoi tho, khong phai dinh vi that.
print('x_impact distinct:', diag_df['x_impact'].nunique(),
      '| y_impact distinct:', diag_df['y_impact'].nunique())
print('Top gia tri x_impact:', Counter(diag_df['x_impact'].dropna()).most_common(5))
print('Top gia tri y_impact:', Counter(diag_df['y_impact'].dropna()).most_common(5))

# Khoang cach trung binh giua diem doan o frame va cham va frame PRE-COLLISION.
# Gan 0 => model bo qua noi dung anh, chi tra ve prior.
# astype(float) la can thiet: cot co the mang dtype=object vi tung chua None.
_d = diag_df.dropna().astype({'x_impact': float, 'y_impact': float,
                              'x_pre': float, 'y_pre': float})
print(f'So video so sanh duoc: {len(_d)}/{len(diag_df)}')
if len(_d):
    print('Khoang cach TB impact vs pre-collision:',
          round(float(np.hypot(_d['x_impact'] - _d['x_pre'],
                               _d['y_impact'] - _d['y_pre']).mean()), 4))
```

Cách đọc kết quả:

| Quan sát | Chẩn đoán | Hướng sửa |
|---|---|---|
| Khoảng cách impact-vs-pre ≈ 0, và ít giá trị phân biệt | Model trả về prior, không đọc ảnh | Đổi **cách hỏi** (xem 6.2, 6.3) |
| Khoảng cách lớn, nhưng S vẫn thấp | Model đọc ảnh nhưng chỉ sai vị trí | Tinh chỉnh **độ phân giải / crop / hậu xử lý** (xem 6.4, 6.5) |
| Nhiều `None` | Lỗi parse output | Sửa **format prompt / regex** (xem 6.6) |

### 6.2 Hướng A — hỏi bằng bounding box thay vì point

Bài học lớn nhất từ `02_type_C`: model bị collapse khi phải chọn từ **tập giá trị neo rời
rạc**. Toạ độ point trên thang [0, 1000] cũng là một dạng như vậy — model có xu hướng trả về
các số "đẹp" (500, 512, 480...).

Bounding box có thể khoẻ hơn vì Qwen3-VL được huấn luyện nặng cho detection, và tâm box là
một đại lượng **suy ra** chứ không phải một số model phải chọn trực tiếp:

```python
GROUNDING_BOX_PROMPT = (
    'This CCTV frame shows a traffic accident. '
    'Detect the vehicles involved in the collision and output the bounding box that '
    'tightly encloses the point of impact between them.\n'
    'Output ONLY this JSON: {"bbox_2d": [x1, y1, x2, y2]}\n'
    'Coordinates are on a 0-1000 scale where x is horizontal (0=left edge, '
    '1000=right edge) and y is vertical (0=top edge, 1000=bottom edge).'
)


def stage3_grounding_v2(video_path, t_final):
    """Grounding qua bbox roi lay tam, thay vi hoi point truc tiep.

    Gia thuyet: bbox la dinh dang Qwen3-VL duoc train nang nhat (detection), va
    tam box la gia tri SUY RA nen khong bi collapse ve cac so 'dep' nhu 500/512
    theo kieu prior ma 02_type_C do duoc cho nhan roi rac.
    """
    frames = sample_frames_stamped(video_path, 1.0, t_final, t_final + 1e-3, limit=1,
                                   max_side=VLM_CFG['grounding_max_side'], burn=False)
    if not frames:
        return None
    j = _extract_json(vlm_generate_oom_safe(GROUNDING_BOX_PROMPT, frames, 64, min_frames=1))
    if not j:
        return None
    box = j.get('bbox_2d') or j.get('bbox') or j.get('box')
    if not (isinstance(box, (list, tuple)) and len(box) >= 4):
        return None
    x1, y1, x2, y2 = (_safe_float(v, np.nan) for v in box[:4])
    if any(np.isnan(v) for v in (x1, y1, x2, y2)):
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if cx > 1.0 or cy > 1.0:                 # thang native [0,1000]
        cx, cy = cx / 1000.0, cy / 1000.0
    return (_clip01(cx), _clip01(cy))
```

### 6.3 Hướng B — bỏ phiếu trên nhiều frame quanh `t_final`

`stage3_grounding` hiện dùng **đúng 1 frame**. Nếu `t_final` lệch 0.5 s, frame đó có thể là
trước hoặc sau va chạm. Lấy 3 frame (`t_final − 0.3`, `t_final`, `t_final + 0.3`), grounding
từng frame rồi lấy **trung vị** (median, không phải mean — median chịu outlier tốt hơn):

```python
def stage3_grounding_v3(video_path, t_final, offsets=(-0.3, 0.0, 0.3)):
    """Grounding tren nhieu frame quanh t_final roi lay trung vi.

    Ly do dung median chu khong mean: mot lan grounding truot han sang xe khac se
    keo mean di rat xa, con median thi khong. Chi phi 3x so ban goc nhung Stage 3
    chi 1 frame/lan nen van re.
    """
    pts = []
    for dt in offsets:
        pt = stage3_grounding(video_path, max(0.0, t_final + dt))
        if pt is not None:
            pts.append(pt)
    if not pts:
        return None
    arr = np.asarray(pts, dtype=float)
    return (_clip01(float(np.median(arr[:, 0]))), _clip01(float(np.median(arr[:, 1]))))
```

Biến thể đáng thử: thay vì median, lọc bỏ điểm cách trung vị > 0.15 rồi lấy mean của phần còn
lại. Và hãy đo **độ tán của 3 điểm** như một tín hiệu tin cậy: nếu 3 lần grounding cho 3 vị
trí rất khác nhau, có lẽ nên rơi về hằng số `CONST` cho video đó.

### 6.4 Hướng C — coarse-to-fine (crop rồi hỏi lại)

Hỏi một lần để lấy vùng thô, crop quanh vùng đó, resize lên rồi hỏi lại trên ảnh crop, cuối
cùng map toạ độ về hệ khung gốc. Đây là kỹ thuật đã có tiền lệ trong repo: notebook 01 dùng
coarse-to-fine cho trục thời gian và đưa T từ 0.23 lên 0.4385.

Lưu ý khi map ngược toạ độ: nếu crop là `[cx0, cy0, cx1, cy1]` (đơn vị chuẩn hoá) và
grounding trên ảnh crop trả về `(u, v)` ∈ [0,1], thì toạ độ khung gốc là
`x = cx0 + u × (cx1 − cx0)`, `y = cy0 + v × (cy1 − cy0)`. Sai bước này là lỗi âm thầm khó
phát hiện nhất trong cả hướng — hãy tự kiểm chứng bằng cách crop toàn khung
(`[0,0,1,1]`) và xác nhận kết quả trùng khít với `stage3_grounding` gốc.

### 6.5 Hướng D — bbox snap bằng YOLO (hậu xử lý)

Writeup của đội hạng 1 ghi nhận detection chỉ còn giá trị như một **bước snap hậu xử lý**
(+0.0005/+0.0013 — nhỏ, nhưng dương). YOLOv8s đã được core load sẵn (`yolo_model`,
`VEHICLE_CLASS_IDS = {2, 3, 5, 7}`). Ý tưởng: sau khi có `(x, y)` từ grounding, chạy YOLO trên
frame tại `t_final`, tìm xe gần nhất, và **kéo điểm về tâm/biên của box đó nếu khoảng cách
nhỏ hơn ngưỡng**. Cẩn thận: nếu kéo vô điều kiện, một grounding đúng có thể bị kéo sang xe
sai. Chỉ snap khi khoảng cách dưới ngưỡng (ví dụ 0.05 đường chéo khung).

### 6.6 Nếu thấy nhiều `None` — kiểm tra output thô trước khi làm gì khác

Trước khi tối ưu prompt, hãy **in ra text thô** model trả về. Rất nhiều "cải tiến" thực chất
chỉ là sửa lỗi parse:

```python
# [DEBUG] In output THO cua grounding cho 5 video dau -- lam viec nay TRUOC khi
# toi uu prompt. Nhieu 'cai tien' hoa ra chi la sua loi parse.
for r in rows_core[:5]:
    vp, t_final = pathlib.Path(r['path']), float(r['accident_time'])
    frames = sample_frames_stamped(vp, 1.0, t_final, t_final + 1e-3, limit=1,
                                   max_side=VLM_CFG['grounding_max_side'], burn=False)
    raw = vlm_generate_oom_safe(GROUNDING_PROMPT, frames, 64, min_frames=1)
    print(f'{vp.stem:42s} t={t_final:6.2f}s -> {raw!r}')
```

### 6.7 Ba cái bẫy đã có tiền lệ trong repo — đừng lặp lại

1. **Tinh chỉnh trên tập lệch.** Một bản trước đã tune trên tập chỉ có `head-on`, đạt 0.80
   ở đó và 0.10 trên tập đa dạng. Luôn đánh giá trên đủ 20 video của `diverse_videos`.
2. **Fallback về model yếu hơn hằng số.** Khi grounding trả `None`, đừng rơi về một model
   khác yếu hơn. OWLv2 + optical flow đạt S = 0.156, thua constant floor 0.22 — rơi về nó
   là **trừ điểm** trên mọi video nó chạm vào. Rơi về `CONST` hoặc về toạ độ Stage 1.
3. **Nhầm hướng của lợi ích.** Với trung bình điều hoà, thành phần yếu nhất mới quan trọng.
   Nếu S của bạn tăng nhưng T hoặc C giảm (vì bạn vô tình đổi `t_final`), điểm tổng có thể
   **giảm**. Patch B in luôn `ACCS` trước/sau để bạn kiểm tra.

---

## Phần 7 — Xử lý sự cố

### 7.1 Lỗi khi nạp core (cell 2)

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `[ERROR] Could not auto-detect the dataset root under /kaggle/input` | Chưa add competition input, hoặc chưa Join Competition | Add Input → Competition → `accident`. Phải đồng ý rules trước. |
| Bảng path audit có dòng `exists=False` | Dataset mount ở layout khác | Đọc log `Contents of /kaggle/input` mà core in ra, rồi set `BASE_DIR` tay trong core cell 6 |
| `install exit code: 1` ở ultralytics/CLIP | Internet chưa bật | Settings → Internet → On (cần xác thực SĐT) |
| `qwen3_vl architecture recognized: False` | transformers cũ vẫn còn trong bộ nhớ | **Restart Kernel**, rồi chạy lại từ cell 1. Bản upgrade không có hiệu lực với module đã import. |
| `'cuda_op_ok': False` + `[DIAGNOSIS] ... architecture mismatch` | torch build không có kernel cho GPU này | Core tự thử hạ về cu121. Nếu vẫn fail: Session → **Factory reset**, hoặc đổi Accelerator sang loại khác rồi đổi lại. |
| `[ERROR] JIT reduction still fails: ... libnvrtc-builtins.so` | Thiếu NVRTC runtime | Đây chính là lý do core cell 55/57 tồn tại. Để chúng chạy, rồi **Restart Kernel** và chạy lại từ cell 1. |
| `Linear4bit layers: 0` | `bitsandbytes` không áp dụng được | Model đang ở fp16 (~16 GB) → sẽ OOM. Kiểm tra `bitsandbytes` đã cài (core cell 8) và `DEVICE == 'cuda'`. |
| `RuntimeError: Qwen3-VL is not usable` | Load model thất bại | Đọc dòng `[ERROR] Qwen3-VL unavailable:` ngay phía trên để biết lý do thật |
| `[CANARY]` trả về `''` rồi `AssertionError` | Model load được nhưng không sinh được text | **Không đi tiếp.** Thường là lỗi NVRTC/JIT hoặc VRAM. Restart Kernel + chạy lại. |
| Kernel chết im lặng trong lúc load model | Hết RAM CPU (không phải VRAM) khi tải 16 GB weight | Đảm bảo Accelerator là GPU (RAM cao hơn), đóng các session khác |

### 7.2 Lỗi khi chạy Patch A / cell 9

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `NameError: name 'eval_core' is not defined` | Chưa chạy Patch A | Chèn và chạy [Patch A](#42--patch-a--dựng-eval_core-và-const_eval) sau cell 2 |
| `NameError: name 'stage2_time_refine_numbered' is not defined` | Patch A chưa chạy (nó tạo alias này) | Như trên |
| `NameError: name 'predict_accident_time_ensemble' is not defined` | Bạn đang gọi `run_inference_vlm` của core | Đừng gọi hàm đó. Gọi từng stage như Patch A/B làm. |
| `[WARNING] OOM -- retrying with 16 frames` xuất hiện ở **mọi** video | Frame budget quá lớn cho VRAM còn lại | Trong cell **riêng của bạn** (không sửa core), hạ config trước khi chạy: `VLM_CFG['whole_limit'] = 16` và/hoặc `VLM_CFG['image_max_side'] = 384`. Ghi rõ vào markdown vì nó làm baseline khác với con số gốc. |
| `[WARNING] OOM persists at 1 frames` | VRAM gần như hết | Restart Kernel; kiểm tra `LOAD_YOLO`/`LOAD_CLIP` (mỗi cái ~0.5 GB); xác nhận `Linear4bit layers` ≠ 0 |
| `RuntimeError: Stage 1 failed to parse JSON on 10 consecutive clips` | Model trả về text không phải JSON | Dùng cell debug ở [6.6](#66-nếu-thấy-nhiều-none--kiểm-tra-output-thô-trước-khi-làm-gì-khác) để xem output thô |
| Cả 20 video ra `xy=(0.500,0.500)` | Stage 1 và Stage 3 đều không parse được | Đây là chế độ lỗi "hằng số đội lốt dự đoán". Debug output thô trước khi tin bất cứ số nào. |
| `AssertionError: Some calibration videos are missing on disk` | Đường dẫn video không resolve được | Kiểm tra `resolve_video_path` và layout `sim_dataset/videos/<type>/` |
| Vòng lặp chạy > 2 giờ | Đang chạy lại toàn bộ 4 stage cho mỗi thí nghiệm | Dùng [Patch B](#43--patch-b--vòng-lặp-đánh-giá-s-nhanh-hơn-15-lần) |

### 7.3 Lệnh chẩn đoán nhanh khi nghi VRAM

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

## Phần 8 — Checklist trước khi mở PR

Theo README **Bước 3** và **Bước 6**:

- [ ] `S` mới **cao hơn cả** `eval_core['S'].mean()` **và** `const_eval['S'].mean()`.
- [ ] Chênh lệch đủ lớn để không phải nhiễu. Với n = 20, tôi khuyến nghị mốc **+0.03 trở lên**,
      và kiểm tra bảng per-video của Patch B để chắc cải thiện là hệ thống chứ không do 1–2 video.
- [ ] `ACCS` tổng **không giảm** (Patch B đã in sẵn trước/sau).
- [ ] Hàm mới có tên riêng (`stage3_grounding_v2`/`v3`/...), **không** ghi đè
      `stage3_grounding` của core.
- [ ] **Không có thay đổi nào trong `core/`** trong diff của bạn. Kiểm tra:
      `git diff --name-only main...feature/spatial-S` — chỉ nên thấy
      `notebooks/03_spatial_S.ipynb` (và file `.md` này nếu bạn cập nhật).
- [ ] Đã ghi bảng kết quả vào markdown cell **đầu** notebook, để người review không phải
      chạy lại mới biết số:

```markdown
## Kết quả (cập nhật lần cuối: 2026-xx-xx)
| Phiên bản | S | ACCS | Ghi chú |
|---|---|---|---|
| Baseline core (`stage3_grounding`) | 0.xxxx | 0.xxxx | 1 frame @768px, hỏi point [0,1000] |
| Constant floor (0.51, 0.51) | 0.xxxx | 0.xxxx | mốc phải vượt |
| `stage3_grounding_v2` (bbox center) | 0.xxxx | 0.xxxx | +0.xxxx |
```

- [ ] Đã lưu output của notebook (bao gồm biểu đồ cell 4 và log Patch B) trước khi commit —
      review sẽ dựa vào đó.
- [ ] Nếu bạn buộc phải sửa core (ví dụ [Patch C](#44--patch-c-tuỳ-chọn--vô-hiệu-hoá-cell-hạ-cấp-torch-của-core)):
      tách sang branch riêng `core/<mô-tả-ngắn>`, PR riêng, tag cả 3 người, và
      **export lại `.py`**: `jupyter nbconvert --to script core/shared_pipeline.ipynb --output shared_pipeline`

---

## Phụ lục — Bảng tra cứu nhanh các hàm core hữu ích cho bạn

| Hàm / biến | Ở core cell | Dùng để làm gì |
|---|---|---|
| `sample_frames_stamped(video, fps, t_start, t_end, limit, max_side, burn)` | 53 | Trích frame, tuỳ chọn đốt timestamp. Trả `[(t, PIL.Image), ...]` |
| `vlm_generate(prompt, images, max_new_tokens)` | 53 | Gọi VLM trực tiếp (greedy). **Không** có retry OOM |
| `vlm_generate_oom_safe(prompt, frames, max_new_tokens, min_frames)` | 53 | Bản có retry OOM (giảm nửa số frame). **Luôn dùng bản này trong vòng lặp** |
| `_extract_json(text)` | 53 | Bắt `{...}` đầu tiên trong text, trả `dict` hoặc `None` |
| `_safe_float(v, default)` / `_clip01(v)` | 59 | Ép float an toàn / kẹp về [0, 1] |
| `normalize_prediction(pred, duration)` | 59 | Ép JSON thô về một dòng submission hợp lệ |
| `extract_frames_window(video, t_center, t_before, t_after, n_frames, max_side)` | 36 | Trích frame quanh một mốc thời gian (không đốt timestamp) |
| `compute_flow_magnitude_map(video, ...)` | 35 | Bản đồ độ lớn optical flow tích luỹ (Farneback) — hữu ích cho hướng lai ghép |
| `analyze_video_collision(video)` | 49 | Pipeline tracking đầy đủ; trả `center` từ giao của 2 box tại impact |
| `yolo_model`, `VEHICLE_CLASS_IDS` | 42 | YOLOv8s đã load; `{2, 3, 5, 7}` = car/motorcycle/bus/truck |
| `spatial_score(px, py, gx, gy)` | 67 | Điểm S của một dự đoán, σ_x = 0.0952, σ_y = 0.1353 |
| `score_predictions(df_pred, labels_ref)` | 67 | Join theo `video_stem`, trả DataFrame có cột `T`, `S`, `C` |
| `accident_score(eval_df)` | 67 | Trung bình điều hoà của 3 trung bình |
| `CONST` | 50 | `{'accident_time': 6.15, 'center_x': 0.51, 'center_y': 0.51, 'type': 'rear-end'}` |
| `VLM_CFG` | 52 | Dict cấu hình frame budget; `grounding_max_side = 768` là của Stage 3 |
| `diverse_videos` / `diverse_labels_df` | 68 | 20 video calibration + ground truth |
| `SIGMA_X`, `SIGMA_Y` | 50 | 0.0952 / 0.1353 — σ của hàm điểm S |

---

## Tham khảo

- Kaggle: ACCIDENT @ CVPR 2026 — AUTOPILOT Workshop
- Writeup hạng 1: arXiv:2605.29325 (Qwen3-VL, 3 lần gọi/clip, không train, 0.57080).
  Stage 3 grounding tách riêng là cải tiến đơn lẻ lớn nhất của họ: **+0.09356**
- Zhao et al., ICML 2021 — *Calibrate Before Use: Improving Few-Shot In-Context Learning
  in Language Models* (phương pháp đo content-free prior mà `02_type_C` đã áp dụng)
- `README.md` của repo — quy ước branch, quy trình PR, quy tắc không sửa core








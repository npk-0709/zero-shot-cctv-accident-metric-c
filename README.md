# Zero-Shot CCTV Accident Analysis (Qwen3-VL & Multimodal Pipeline)

Hệ thống phân tích tai nạn giao thông trên camera CCTV theo phương pháp Zero-Shot / Multimodal 3-stage pipeline (Temporal, Collision Type, Spatial Grounding) sử dụng Qwen3-VL, YOLO, CLIP và kinematics tracking.

---

## Cấu trúc Repository

```text
zero-shot-cctv-accident/
├── core/
│   ├── shared_pipeline.ipynb   # Pipeline hoàn chỉnh baseline (T, S, C)
│   └── shared_pipeline.py      # Python module chuyển đổi từ notebook core để import
├── notebooks/
│   ├── 01_temporal_T.ipynb     # Thử nghiệm & tối ưu stage 1 (Temporal - T): NumPro-style, coarse-to-fine
│   ├── 02_type_C.ipynb         # Thử nghiệm & tối ưu stage 2 (Collision Type - C): Bias calibration, crop/zoom prompt
│   └── 03_spatial_S.ipynb      # Thử nghiệm & tối ưu stage 3 (Spatial - S): Optical flow centroid & grounding
├── legacy/
│   └── classical_pipeline.ipynb # Object-centric pipeline cũ (SORT tracker + CLIP) dùng làm baseline so sánh
└── README.md
```

---

## Thiết kế Kiến trúc & Tách biệt Metric

Do 3 metric Temporal (T), Collision Type (C), và Spatial Grounding (S) gọi chung VLM 3-stage cascade nối tiếp nhau cho mỗi video:
1. `core/shared_pipeline.py` chứa toàn bộ hạ tầng chung (Data loading, YOLO, CLIP, Qwen3-VL loading, track kinematics, 3 stage functions baseline, run_inference_vlm, scoring evaluation).
2. Các notebook chuyên biệt tại `notebooks/` sẽ import `shared_pipeline.py` ở cell đầu tiên để có điểm số baseline nền, sau đó chỉ override/patch đúng 1 stage function để thử nghiệm cải tiến mà không làm nhiễu điểm số của 2 stage còn lại.

---

## Hướng dẫn Sử dụng & Chạy Notebook

### 1. Trên Chạy Local / Server có GPU
Mỗi notebook chuyên biệt bắt đầu với cell import tự động:
```python
import sys, os
sys.path.append('../core')
%run ../core/shared_pipeline.py
```

### 2. Trên Kaggle / Google Colab (Notebook Độc lập)
Nếu chạy notebook trực tiếp trên Kaggle/Colab mà không mount toàn bộ repo, có thể tải file `shared_pipeline.py` về môi trường bằng lệnh:
```python
!curl -s -o shared_pipeline.py https://raw.githubusercontent.com/HoangDinhBui/zero-shot-cctv-accident/main/core/shared_pipeline.py
%run shared_pipeline.py
```

---

## Phát triển song song — 3 người, mỗi người 1 metric

Mục tiêu: mỗi người chỉ động vào đúng 1 notebook (`01_temporal_T`, `02_type_C`, hoặc `03_spatial_S`), không đụng vào `core/` trừ khi thực sự cần, để không ai bị conflict hay bị điểm số của mình nhiễu bởi thay đổi của người khác.

### Bước 1 — Clone và tạo branch riêng

```bash
git clone https://github.com/HoangDinhBui/zero-shot-cctv-accident.git
cd zero-shot-cctv-accident
git checkout -b feature/temporal-T      # người phụ trách T
git checkout -b feature/type-C          # người phụ trách C
git checkout -b feature/spatial-S       # người phụ trách S
```

Quy ước tên branch: `feature/<metric>-<X>` với `<X>` là chữ metric viết hoa (`T`/`C`/`S`) để dễ lọc trong danh sách PR.

### Bước 2 — Xác nhận baseline chạy được trước khi sửa gì

Trước khi thử nghiệm, mỗi người chạy hết notebook của mình **không sửa gì** để chắc `core/shared_pipeline.py` đang cho điểm baseline đúng như ghi trong README (mục Evaluation). Nếu baseline không khớp, báo lại trong group trước khi tiếp tục — đừng vừa sửa core vừa sửa notebook chuyên biệt cùng lúc, sẽ không biết điểm thay đổi vì đâu.

### Bước 3 — Chỉ override đúng 1 stage function, không sửa core

Trong notebook của mình, **không sửa trực tiếp** function baseline đã import từ `shared_pipeline.py` (`stage1_full_scan`, `stage3_grounding`, `classify_type_cascade`). Thay vào đó, định nghĩa lại bằng tên mới trong chính notebook chuyên biệt, theo đúng convention `_v2`/`_v3` đã dùng xuyên suốt repo, rồi patch `run_inference_vlm` để trỏ vào bản mới — 2 stage còn lại giữ nguyên baseline từ core:

```python
# Vi du: nguoi phu trach T dinh nghia ham moi trong 01_temporal_T.ipynb,
# KHONG dong vao stage1_full_scan goc trong core.
def stage1_full_scan_v2(video_path, duration, scene):
    ...

# Patch lai pipeline: chi thay Stage 1, Stage 2 (C) va Stage 3 (S) van la ban
# baseline tu core -- de diem C/S do duoc trong notebook nay khong bi nhieu
# boi thay doi cua ban.
def run_inference_vlm_T_experiment(video_path, ...):
    pred = stage1_full_scan_v2(video_path, duration, scene)   # <- ban moi
    t_final = stage2_time_refine_numbered(video_path, pred['accident_time'], duration)
    type_pred = classify_type_cascade(video_path, t_final, pred['type'])   # baseline core
    spatial_pred = stage3_grounding(video_path, t_final)                   # baseline core
    ...
```

Cuối notebook, luôn có 1 cell so sánh **trước/sau** trên `diverse_videos` (bộ 20 video calibration) và ghi lại số liệu ngay trong 1 markdown cell ở đầu notebook (để review PR không phải chạy lại mới biết kết quả):

```markdown
## Kết quả (cập nhật lần cuối: <ngày>)
Baseline T (core): 0.4385
Sau thử nghiệm    : 0.xxxx  (+/- x.xx)
```

### Bước 4 — Nếu bắt buộc phải sửa `core/`

Chỉ sửa `core/` khi phát hiện thiếu 1 helper dùng chung thật sự cần cho cả 3 người (ví dụ thêm 1 hàm sampling frame mới mà cả T lẫn C đều cần). Khi đó:
1. Mở branch riêng `core/<mo-ta-ngan>`, không sửa trực tiếp trên `feature/*` của mình.
2. Sau khi sửa `shared_pipeline.ipynb`, luôn export lại `.py` trước khi commit — 2 file phải đồng bộ:
```bash
   jupyter nbconvert --to script core/shared_pipeline.ipynb --output shared_pipeline
```
3. Mở PR riêng cho thay đổi core, gắn tag cả 3 người review, vì thay đổi core ảnh hưởng đến baseline của cả 3 notebook.
4. Sau khi PR core được merge vào `main`, cả 3 người `git pull origin main` vào branch của mình và chạy lại bước 2 (xác nhận baseline) trước khi tiếp tục — baseline có thể đã đổi.

### Bước 5 — Đồng bộ trong lúc làm

```bash
git fetch origin
git rebase origin/main      # keo thay doi core moi nhat (neu co) vao branch minh
```
Vì mỗi người chỉ sửa 1 notebook riêng (`01_*`, `02_*`, hoặc `03_*`), rebase gần như không bao giờ conflict trừ khi ai đó lỡ tay sửa `core/`.

### Bước 6 — Mở PR và tích hợp cuối cùng

Mỗi người mở PR từ branch của mình vào `main`, mô tả PR nêu rõ điểm số trước/sau (đúng bảng markdown ở bước 3). Sau khi cả 3 PR merge riêng lẻ (không ai đụng core, nên merge độc lập, không cần chờ nhau):

- **Bắt buộc chạy lại 1 lần đánh giá tổng hợp** trên `core/shared_pipeline.ipynb` sau khi gộp cả 3 cải tiến vào cùng 1 `run_inference_vlm` — vì 3 stage chạy nối tiếp (T → C → S dùng chung `t_final` từ Stage 1), cải tiến T có thể làm thay đổi input đầu vào của C và S theo hướng không lường trước được khi đo riêng lẻ. Điểm tổng T+S+C sau khi gộp không nhất thiết bằng tổng 3 cải tiến đo độc lập.
- Người tổng hợp (rotate giữa 3 người mỗi lần) chịu trách nhiệm cập nhật `core/shared_pipeline.ipynb` với 3 stage function tốt nhất, export lại `.py`, và cập nhật bảng điểm trong mục Evaluation của README này.

---

## Evaluation & Benchmark
Pipeline đánh giá dựa trên bộ Diverse Calibration Set gồm 20 video (4 video cho mỗi loại va chạm `t-bone`, `rear-end`, `head-on`, `sideswipe`, `single-vehicle`).
- Score Temporal (T): Gaussian scoring curve xung quanh timestamp chính xác.
- Score Spatial (S): Euclidean distance từ điểm va chạm dự đoán tới ground-truth centroid.
- Score Type (C): Top-1 accuracy và weighted F1-score cho các phân loại tai nạn.

---

## References
- Zhao et al. ICML 2021: Calibrate Before Use: Improving Few-Shot In-Context Learning in Language Models
- Qwen3-VL: Vision-Language Model backbone for multimodal temporal & visual grounding.
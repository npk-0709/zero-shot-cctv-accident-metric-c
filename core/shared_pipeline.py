#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Shared Pipeline Core Module - Auto-extracted


# ===========================================================================
# # ACCIDENT @ CVPR: Zero-Shot CCTV Traffic Accident Understanding
# 
# **Competition:** ACCIDENT @ CVPR 2026 -- AUTOPILOT Workshop (Kaggle)
# 
# **Notebook Focus:**
# - Temporal localization of accident onset from fixed-view CCTV video clips
# - Spatial localization of impact point via open-vocabulary grounding and optical flow
# - Zero-shot collision-type classification using pre-trained vision-language models
# ===========================================================================


# ===========================================================================
# ## Introduction
# 
# Traffic accident analysis from fixed CCTV infrastructure presents challenges distinct from general video understanding. Accident events are temporally sparse, spatially unpredictable, and visually subtle relative to normal traffic flow. The benchmark defined by ACCIDENT @ CVPR 2026 compounds these challenges by excluding labeled real training footage, requiring methods that generalize without dataset-specific fine-tuning.
# 
# This notebook constructs a zero-shot inference pipeline whose **primary signal is per-vehicle detection and tracking**: an accident is an event between two specific objects, so the colliding pair's kinematics (contact time, joint deceleration, box-intersection centroid, approach-angle geometry) answer all three benchmark questions more directly than any whole-frame signal can. A chain of zero-shot vision-language models backs every stage up when tracking is unreliable:
# 
# | Stage | Question | Primary signal (tracking) | Fallback (zero-shot) |
# |-------|----------|---------------------------|----------------------|
# | 1 -- *When* | accident time | first box contact + joint deceleration peak | frame-diff anchor + bounded PE refinement |
# | 2 -- *What* | collision type | pre-impact velocity angle + contact geometry | CLIP + Qwen2.5-VL soft ensemble |
# | 3 -- *Where* | impact point | box-intersection centroid at impact | OWLv2 grounding + flow centroid |
# 
# The synthetic CARLA dataset is used for pipeline calibration and EDA. Submissions target the real CCTV test set. Every design decision below is validated (or rejected) on a *diverse* calibration set covering all five collision types -- an earlier revision tuned on a head-on-only subset and regressed badly on the other four types (see Section 7).
# ===========================================================================


# ===========================================================================
# ## Table of Contents
# 
# 1. [Data Acquisition](#1-data-acquisition)
# 2. [Data Inspection](#2-data-inspection)
# 3. [Data Cleaning](#3-data-cleaning)
# 4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
# 5. [Feature Engineering](#5-feature-engineering)
# 6. [Modeling](#6-modeling)
# 7. [Evaluation](#7-evaluation)
# 8. [Test Inference and Submission](#8-test-inference-and-submission)
# 9. [Conclusion](#9-conclusion)
# 10. [References](#10-references)
# ===========================================================================


# ===========================================================================
# ---
# ## 0. Environment Setup
# ===========================================================================


# --- Code Cell ---
# [SETUP] Standard library imports -- order: stdlib, third-party, local
import re
import json
import gzip
import time
import warnings
import pathlib
from collections import defaultdict

# [SETUP] Numerical and data processing
import numpy as np
import pandas as pd

# [SETUP] Visualization stack
import matplotlib.pyplot as plt
import seaborn as sns

# [SETUP] Computer vision
import cv2
from PIL import Image as PILImage

# [SETUP] Suppress non-critical runtime warnings
warnings.filterwarnings('ignore')

print('[STATUS] Core imports complete')


# --- Code Cell ---
# [SETUP] Global matplotlib and seaborn configuration
sns.set_theme(style='whitegrid', context='notebook')

plt.rcParams.update({
    'figure.figsize'  : (10, 6),
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 11,
    'xtick.labelsize' : 10,
    'ytick.labelsize' : 10,
    'legend.fontsize' : 10,
    'figure.dpi'      : 120,
    'savefig.bbox'    : 'tight',
})

# [SETUP] Consistent color palette for all plots
PALETTE = {
    'primary'    : '#1f77b4',
    'secondary'  : '#ff7f0e',
    'tertiary'   : '#2ca02c',
    'quaternary' : '#d62728',
    'quinary'    : '#9467bd',
    'senary'     : '#8c564b',
}

print('[STATUS] Plot configuration applied')


# --- Code Cell ---
# [SETUP] Reproducibility seed for all stochastic operations
SEED = 42
np.random.seed(SEED)

# [SETUP] Root paths -- auto-detected, do not hardcode.
#
# Kaggle mounts data at a DIFFERENT prefix depending on how it was attached:
#   - competition data (Add Input -> Competition) -> /kaggle/input/competitions/<slug>/
#   - a regular Dataset (Add Input -> Dataset)     -> /kaggle/input/<slug>/
# A previous revision hardcoded the Dataset-style path
# (/kaggle/input/accident) with a comment noting the competition-style path
# was the correct one -- and never applied its own fix. Every downstream
# exists() check then silently returned False and produced 0 videos. Probe
# both known layouts, and if neither matches, search all of /kaggle/input for
# any directory that actually contains a sim_dataset/ or videos/ folder, so
# this survives being forked, renamed, or re-attached differently.
_CANDIDATE_ROOTS = [
    pathlib.Path('/kaggle/input/competitions/accident'),
    pathlib.Path('/kaggle/input/accident'),
]

def _looks_like_dataset_root(p: pathlib.Path) -> bool:
    return p.is_dir() and ((p / 'sim_dataset').exists() or (p / 'videos').exists())

BASE_DIR = next((r for r in _CANDIDATE_ROOTS if _looks_like_dataset_root(r)), None)

if BASE_DIR is None:
    _input_root = pathlib.Path('/kaggle/input')
    if _input_root.exists():
        for _entry in sorted(_input_root.iterdir()):
            if _looks_like_dataset_root(_entry):
                BASE_DIR = _entry
                break
            # competition-style datasets can nest one level deeper, e.g.
            # /kaggle/input/competitions/<slug>/
            if _entry.is_dir():
                for _sub in sorted(_entry.iterdir()):
                    if _looks_like_dataset_root(_sub):
                        BASE_DIR = _sub
                        break
            if BASE_DIR is not None:
                break

if BASE_DIR is None:
    print('[ERROR] Could not auto-detect the dataset root under /kaggle/input.')
    print('        Attach the "accident" competition data via Add Input, or if it is')
    print('        already attached, verify the actual layout below and set BASE_DIR')
    print('        manually in this cell.')
    _input_root = pathlib.Path('/kaggle/input')
    if _input_root.exists():
        print(f'[STATUS] Contents of {_input_root}:')
        for _entry in sorted(_input_root.iterdir()):
            print(f'    {_entry}')
    else:
        print('[STATUS] /kaggle/input does not exist -- not running on Kaggle, or no data attached.')
    BASE_DIR = pathlib.Path('/kaggle/input/accident')  # placeholder so later cells don't NameError

SYNTHETIC_DIR   = BASE_DIR / 'sim_dataset'    # synthetic CARLA dataset
REAL_VIDEOS_DIR = BASE_DIR / 'videos'         # real CCTV test videos
OUTPUT_DIR      = pathlib.Path('/kaggle/working')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# [SETUP] Key file paths
LABELS_CSV            = SYNTHETIC_DIR / 'labels.csv'                # synthetic supervision index
TEST_METADATA_CSV     = BASE_DIR      / 'test_metadata.csv'         # real test scene tags
SAMPLE_SUBMISSION     = BASE_DIR      / 'sample_submission.csv'     # output schema reference
ANNOTATION_CLASSES    = SYNTHETIC_DIR / 'annotation_classes.yaml'   # segmentation class map
SYNTHETIC_VIDEOS_DIR  = SYNTHETIC_DIR / 'videos'                    # {head-on,rear-end,sideswipe,single,t-bone}/
VIDEO_ANNOTATIONS_DIR = SYNTHETIC_DIR / 'video_annotations'         # per-video annotation files

COLLISION_TYPES = ['head-on', 'rear-end', 'sideswipe', 'single', 't-bone']


def resolve_video_path(rgb_path: str) -> pathlib.Path:
    """Resolve an rgb_path value from labels.csv to an absolute video path.

    rgb_path values are relative to sim_dataset/ (e.g. videos/head-on/clip.mp4),
    but earlier dataset revisions used paths relative to the competition root.
    Try both roots and return whichever exists, so downstream code never
    silently operates on a non-existent path (an earlier revision of this
    notebook lost 20/20 calibration videos to exactly that bug).
    """
    p = pathlib.Path(rgb_path)
    if p.is_absolute():
        return p
    for root in (SYNTHETIC_DIR, BASE_DIR):
        candidate = root / p
        if candidate.exists():
            return candidate
    return SYNTHETIC_DIR / p  # best guess; existence is asserted before inference


print('[STATUS] Path constants initialized')
print(f'  BASE_DIR        : {BASE_DIR}')
print(f'  SYNTHETIC_DIR   : {SYNTHETIC_DIR}')
print(f'  REAL_VIDEOS_DIR : {REAL_VIDEOS_DIR}')


# --- Code Cell ---
# [STATUS] File availability audit -- all critical paths verified before downstream steps
path_checks = [
    ('BASE_DIR',              BASE_DIR),
    ('SYNTHETIC_DIR',         SYNTHETIC_DIR),
    ('SYNTHETIC_VIDEOS_DIR',  SYNTHETIC_VIDEOS_DIR),
    ('VIDEO_ANNOTATIONS_DIR', VIDEO_ANNOTATIONS_DIR),
    ('REAL_VIDEOS_DIR',       REAL_VIDEOS_DIR),
    ('labels.csv',            LABELS_CSV),
    ('annotation_classes.yaml', ANNOTATION_CLASSES),
    ('test_metadata.csv',     TEST_METADATA_CSV),
    ('sample_submission.csv', SAMPLE_SUBMISSION),
]

audit_df = pd.DataFrame([
    {
        'label'  : label,
        'path'   : str(p),
        'exists' : p.exists(),
        'kind'   : 'dir' if p.is_dir() else ('file' if p.is_file() else 'missing'),
    }
    for label, p in path_checks
])

# All rows should show exists=True before proceeding
display(audit_df.style.map(
    lambda v: 'color: green; font-weight: bold' if v is True
              else ('color: red; font-weight: bold' if v is False else ''),
    subset=['exists']
))


# --- Code Cell ---
# [SETUP] Install model dependencies (requires internet on Kaggle)
#   - ultralytics    : YOLOv8 vehicle detector (tracking stage)
#   - CLIP           : zero-shot classification (Stage 2)
#   - perception_models : Perception Encoder for temporal refinement (Stage 1)
#   - transformers stack : Qwen2.5-VL (Stage 2) and OWLv2 (Stage 3)
import subprocess
import sys
import json as _json


def _sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def _torch_smoke_test():
    """Import torch and run ONE real op on cuda, in a fresh subprocess.

    torch.cuda.is_available() returning True is not sufficient evidence the
    GPU actually works: a torch build whose compiled kernels don't cover this
    GPU's compute capability still reports CUDA as "available" and only fails
    once a real kernel launch is attempted. Also captures get_device_name /
    get_device_capability / get_arch_list, so an architecture mismatch is
    diagnosed directly instead of guessed at from the exception text alone.
    Runs in a subprocess -- not `import torch` in this cell -- because once
    this kernel process has torch in sys.modules, a later pip install that
    replaces the on-disk package would not change what an already-imported
    `torch` in this process reports.
    """
    code = (
        "import torch, json\n"
        "info = {'version': torch.__version__, 'cuda_available': torch.cuda.is_available()}\n"
        "try:\n"
        "    if info['cuda_available']:\n"
        "        info['device_name'] = torch.cuda.get_device_name(0)\n"
        "        info['device_capability'] = list(torch.cuda.get_device_capability(0))\n"
        "        info['torch_arch_list'] = torch.cuda.get_arch_list()\n"
        "        _ = (torch.tensor([1.0, 2.0], device='cuda') * 2).cpu()\n"
        "        _ = torch.prod(torch.tensor([2, 3, 4], dtype=torch.int64, device='cuda')).cpu()\n"
        "        info['cuda_op_ok'] = True\n"
        "    else:\n"
        "        info['cuda_op_ok'] = None\n"
        "except Exception as e:\n"
        "    info['cuda_op_ok'] = False\n"
        "    info['cuda_op_error'] = f'{type(e).__name__}: {e}'\n"
        "print(json.dumps(info))\n"
    )
    rc, out, err = _sh([sys.executable, '-c', code])
    lines = [l for l in out.strip().splitlines() if l.strip()]
    if lines:
        try:
            return _json.loads(lines[-1])
        except _json.JSONDecodeError:
            pass
    return {'version': None, 'cuda_available': None, 'cuda_op_ok': False,
            'raw_stdout': out[-500:], 'raw_stderr': err[-500:]}


def _diagnose_arch_mismatch(info):
    """Compare the GPU's actual compute capability against what this torch
    build shipped kernels for, so the report says WHY, not just THAT it failed."""
    cap = info.get('device_capability')
    arch_list = info.get('torch_arch_list') or []
    if not cap or not arch_list:
        return None
    sm = f'sm_{cap[0]}{cap[1]}'
    compiled_sms = {a.split('_')[0] + '_' + a.split('_')[1][:2] for a in arch_list if 'sm_' in a}
    if sm not in compiled_sms and not any(a.startswith(f'compute_{cap[0]}{cap[1]}') for a in arch_list):
        return (f"GPU is {info.get('device_name')} (compute capability {cap[0]}.{cap[1]}, i.e. {sm}), "
                f"but this torch build only has compiled kernels for: {arch_list}. "
                f"{sm} is NOT in that list -- this is an architecture mismatch, not a driver problem.")
    return None


print('[STATUS] Baseline torch smoke test (before any installs)...')
_baseline = _torch_smoke_test()
print(f'[STATUS] Baseline: {_baseline}')

_repaired = False
if _baseline.get('cuda_op_ok') is False:
    _diag = _diagnose_arch_mismatch(_baseline)
    if _diag:
        print(f'[DIAGNOSIS] {_diag}')
    else:
        print('[DIAGNOSIS] cuda_available=True but a real op fails, and device_capability/'
              'arch_list were not both captured (the failure happened before those calls) -- '
              'consistent with either an architecture mismatch or a driver/runtime version '
              'mismatch between this torch build and the host.')
    print('[WARNING] A real CUDA op already fails before any installs run -- this is a '
          'pre-existing problem with the environment\'s preinstalled torch, not something '
          'the installs below will cause. Attempting a repair: reinstalling torch/torchvision/'
          'torchaudio from the stable cu121 channel, which has shipped sm_75 (T4) kernels '
          'continuously and is a safer bet than whatever cu128 build shipped with this image.')
    _rc, _out, _err = _sh([
        sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/cu121',
    ])
    print(f'[STATUS] Repair install exit code: {_rc}')
    if _rc != 0:
        print(_err[-1500:])
    _baseline = _torch_smoke_test()
    print(f'[STATUS] Post-repair smoke test: {_baseline}')
    if _baseline.get('cuda_op_ok') is True:
        _repaired = True
        print('[SUCCESS] Repair worked -- proceeding with the cu121 build for the rest of this session.')
    else:
        _diag2 = _diagnose_arch_mismatch(_baseline)
        print('[ERROR] Repair did not fix it.', (_diag2 or ''))
        print('        This looks like a Kaggle platform/accelerator issue rather than something')
        print('        pip can fix from inside the notebook. Try: Session -> Factory reset, or')
        print('        switch the accelerator (Settings -> Accelerator) to a different GPU option')
        print('        and back, then Restart Kernel and re-run from the top. Do not proceed to')
        print('        load any model on GPU until this smoke test reports cuda_op_ok=True.')

# Pin torch/torchvision/torchaudio/triton to whatever is on disk RIGHT NOW (post-repair, if a
# repair happened above), so nothing pulled in as a transitive dependency of ultralytics / CLIP /
# perception_models / the transformers stack below is allowed to silently swap them again.
_CONSTRAINTS_PATH = '/kaggle/working/_torch_constraints.txt'
_pinned = []
for _pkg in ('torch', 'torchvision', 'torchaudio', 'triton'):
    _rc, _out, _err = _sh([sys.executable, '-m', 'pip', 'show', _pkg])
    if _rc == 0:
        for _line in _out.splitlines():
            if _line.startswith('Version:'):
                _pinned.append(f"{_pkg}=={_line.split(':', 1)[1].strip()}")
with open(_CONSTRAINTS_PATH, 'w') as f:
    f.write('\n'.join(_pinned) + '\n')
print(f'[STATUS] Pinning during install: {_pinned}')


def pip_install(args, label):
    result = subprocess.run(
        ['pip', 'install', '-q', '-c', _CONSTRAINTS_PATH] + args,
        capture_output=True, text=True)
    print(f'[STATUS] {label} install exit code: {result.returncode}')
    if result.returncode != 0:
        # A constraint conflict means this package genuinely needs a different
        # torch than what's on disk. Surface it now and stop -- silently retrying
        # without -c would defeat the whole point of pinning.
        print(result.stderr[-2000:])
    return result.returncode


pip_install(['ultralytics'], 'ultralytics (YOLOv8)')
pip_install(['git+https://github.com/openai/CLIP.git'], 'CLIP')

result = subprocess.run(
    ['git', 'clone', 'https://github.com/facebookresearch/perception_models.git'],
    capture_output=True, text=True
)
print('[STATUS] perception_models clone exit code:', result.returncode)
pip_install(['-e', 'perception_models'], 'perception_models')

# -U is not optional. Kaggle ships a transformers that predates Qwen3-VL, and
# `pip install transformers` on an already-satisfied requirement is a no-op --
# it exits 0, prints nothing, and leaves the old version in place. That is
# exactly how a previous run reported "install exit code: 0" and then failed
# with "Transformers does not recognize qwen3_vl", silently fell back to
# constants for all 2027 clips, and produced a submission that scored nothing.
pip_install(['-U', 'transformers>=4.57', 'accelerate', 'qwen-vl-utils', 'bitsandbytes'],
            'transformers stack (upgraded for Qwen3-VL)')

print('[STATUS] Post-install torch smoke test...')
_after = _torch_smoke_test()
print(f'[STATUS] After installs: {_after}')

if _baseline.get('version') != _after.get('version'):
    print(f"[ERROR] torch was changed by the installs above despite the -c constraints "
          f"file: {_baseline.get('version')} -> {_after.get('version')}. One of the "
          f"packages above must have forced this through some path the constraints file "
          f"doesn't cover. Do not proceed to load models until this is understood -- "
          f"Restart Kernel and re-run first.")
elif _after.get('cuda_op_ok') is False:
    print(f"[ERROR] torch version is unchanged ({_after.get('version')}) but a real CUDA "
          f"op now fails: {_after.get('cuda_op_error')}. Since the pre-install baseline "
          f"above already {'was fixed by the repair' if _repaired else 'had the same problem'}, "
          f"this is consistent with an unstable environment rather than these installs -- "
          f"do not proceed to Section 6.10 (Qwen3-VL) or CLIP loading below.")
elif _after.get('cuda_op_ok') is True:
    print('[STATUS] torch unchanged (or successfully repaired) and a real CUDA op still '
          'works -- safe to build on.')

import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

print(f'[STATUS] transformers {transformers.__version__}')
QWEN3_VL_OK = 'qwen3_vl' in CONFIG_MAPPING_NAMES
print(f'[STATUS] qwen3_vl architecture recognized: {QWEN3_VL_OK}')
if not QWEN3_VL_OK:
    print('[FATAL] This transformers cannot load Qwen3-VL. Section 6.10 will not run.')
    print('        Restart the kernel after this cell (an upgrade does not take effect')
    print('        in an already-imported module), or install from source:')
    print('          pip install -U git+https://github.com/huggingface/transformers.git')

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[STATUS] torch {torch.__version__} | device: {DEVICE}')



# ===========================================================================
# ---
# ## 1. Data Acquisition
# 
# This section loads all structured metadata from disk: the synthetic labels index, per-video annotation references, and the real test metadata. Raw binary video assets are accessed later via path references rather than bulk loading.
# ===========================================================================


# --- Code Cell ---
# [LOAD] Synthetic labels index -- primary supervision signal for pipeline calibration
labels_df = pd.read_csv(LABELS_CSV) if LABELS_CSV.exists() else pd.DataFrame()

# [LOAD] Real test metadata -- coarse scene tags provided for analysis, not scoring
test_df = pd.read_csv(TEST_METADATA_CSV) if TEST_METADATA_CSV.exists() else pd.DataFrame()

dim_report = pd.DataFrame({
    'dataset' : ['synthetic_labels', 'test_metadata'],
    'rows'    : [len(labels_df), len(test_df)],
    'columns' : [labels_df.shape[1] if not labels_df.empty else 0,
                 test_df.shape[1] if not test_df.empty else 0],
})
display(dim_report)
print('[SUCCESS] Metadata loaded')


# --- Code Cell ---
# [LOAD] Sample submission -- defines expected output schema for the final CSV
sample_sub = pd.read_csv(SAMPLE_SUBMISSION) if SAMPLE_SUBMISSION.exists() else pd.DataFrame()

print('Submission columns:', list(sample_sub.columns))
display(sample_sub.head(3))


# --- Code Cell ---
# [LOAD] Enumerate video files
# Synthetic structure: sim_dataset/videos/{head-on,rear-end,sideswipe,single,t-bone}/*.mp4
synthetic_videos = []
if SYNTHETIC_VIDEOS_DIR.exists():
    for subdir in COLLISION_TYPES:
        synthetic_videos.extend(sorted((SYNTHETIC_VIDEOS_DIR / subdir).glob('*.mp4')))

# Real CCTV test videos: flat directory under videos/
real_videos = sorted(REAL_VIDEOS_DIR.glob('*.mp4')) if REAL_VIDEOS_DIR.exists() else []

inventory_rows = [{'split': 'real_test', 'collision_type': 'all', 'count': len(real_videos)}]
for subdir in COLLISION_TYPES:
    sp = SYNTHETIC_VIDEOS_DIR / subdir
    inventory_rows.append({
        'split': 'synthetic', 'collision_type': subdir,
        'count': len(list(sp.glob('*.mp4'))) if sp.exists() else 0,
    })

display(pd.DataFrame(inventory_rows))
print(f'[STATUS] Total synthetic videos : {len(synthetic_videos)}')
print(f'[STATUS] Total real test videos : {len(real_videos)}')


# --- Code Cell ---
# [LOAD] Annotation files inventory
# video_annotations/ contains subdirs named like Town03_head-on_clear_00.json/
# Each subdir holds the actual annotation file(s) -- recurse and filter to files only
annotation_files = [
    p for p in sorted(VIDEO_ANNOTATIONS_DIR.rglob('*')) if p.is_file()
] if VIDEO_ANNOTATIONS_DIR.exists() else []

ann_subdirs = [
    p for p in sorted(VIDEO_ANNOTATIONS_DIR.glob('*')) if p.is_dir()
] if VIDEO_ANNOTATIONS_DIR.exists() else []

display(pd.DataFrame({
    'metric' : ['annotation_dir_exists', 'annotation_subdirs', 'annotation_files', 'sample_subdir_name'],
    'value'  : [VIDEO_ANNOTATIONS_DIR.exists(), len(ann_subdirs), len(annotation_files),
                ann_subdirs[0].name if ann_subdirs else 'n/a'],
}))

# [LOAD] Inspect schema of the first annotation file (handles .json and .json.gz)
if annotation_files:
    first_ann = annotation_files[0]
    open_fn = gzip.open if first_ann.suffix == '.gz' else open
    with open_fn(first_ann, 'rt', encoding='utf-8') as f:
        ann_sample = json.load(f)
    top_keys = list(ann_sample.keys()) if isinstance(ann_sample, dict) else f'list[{len(ann_sample)}]'
    print(f'[STATUS] {first_ann.name} top-level keys:', top_keys)
else:
    print('[STATUS] No annotation files found')


# ===========================================================================
# ---
# ## 2. Data Inspection
# 
# Systematic audit of all loaded data structures: schema review, null quantification, statistical profiling, and video-level properties (resolution, FPS, duration) extracted from a sample of clips for hardware-aware planning.
# ===========================================================================


# --- Code Cell ---
# [INSPECT] Schema of synthetic labels -- column names, dtypes, null counts
if not labels_df.empty:
    display(pd.DataFrame({
        'column'    : labels_df.columns,
        'dtype'     : labels_df.dtypes.values,
        'null_count': labels_df.isnull().sum().values,
        'null_pct'  : (labels_df.isnull().mean().values * 100).round(2),
    }))


# --- Code Cell ---
# [INSPECT] Statistical summary and sample rows
if not labels_df.empty:
    display(labels_df.describe().T.round(4))
    display(labels_df.head(5))


# --- Code Cell ---
# [INSPECT] Collision type distribution in synthetic labels
if not labels_df.empty and 'type' in labels_df.columns:
    type_counts = labels_df['type'].value_counts().reset_index()
    type_counts.columns = ['collision_type', 'count']
    type_counts['pct'] = (type_counts['count'] / type_counts['count'].sum() * 100).round(2)
    display(type_counts)


# --- Code Cell ---
# [INSPECT] Schema of real test metadata
if not test_df.empty:
    display(pd.DataFrame({
        'column'    : test_df.columns,
        'dtype'     : test_df.dtypes.values,
        'null_count': test_df.isnull().sum().values,
    }))
    display(test_df.head(5))


# --- Code Cell ---
# [INSPECT] Video-level property extraction from a sample of synthetic clips
# Reads container metadata only -- no full video decoding

def extract_video_meta(video_path: pathlib.Path) -> dict:
    """Return FPS, frame count, width, height, duration for a single video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        'path'      : video_path.name,
        'fps'       : round(fps, 2),
        'n_frames'  : n_frames,
        'width'     : width,
        'height'    : height,
        'duration_s': round(n_frames / fps, 2) if fps > 0 else 0.0,
    }

SAMPLE_N = min(20, len(synthetic_videos))
video_meta_df = pd.DataFrame(
    [r for r in (extract_video_meta(p) for p in synthetic_videos[:SAMPLE_N]) if r]
)

if not video_meta_df.empty:
    display(video_meta_df.describe().T.round(2))
else:
    print('[STATUS] No video metadata extracted -- check synthetic video paths')


# ===========================================================================
# ---
# ## 3. Data Cleaning
# 
# Targeted cleaning of the synthetic labels index: duplicate removal, type-label normalization, out-of-bounds coordinate clamping, and target validation. Video paths are resolved with `resolve_video_path` (which checks both candidate roots) so every retained record points at a file that actually exists.
# ===========================================================================


# --- Code Cell ---
# [CLEAN] Work on a copy; drop exact duplicates; normalize type labels
labels_clean = labels_df.copy() if not labels_df.empty else pd.DataFrame()
pre_clean_rows = len(labels_clean)

if not labels_clean.empty:
    labels_clean = labels_clean.drop_duplicates()
    if 'type' in labels_clean.columns:
        labels_clean['type'] = labels_clean['type'].str.strip().str.lower()
    print(f'[STATUS] Rows: {pre_clean_rows} -> {len(labels_clean)} after duplicate removal')
    print('[STATUS] Type values:', sorted(labels_clean['type'].unique()))


# --- Code Cell ---
# [CLEAN] Clamp spatial coordinates to [0, 1]; drop rows with missing or invalid targets
coord_cols       = ['center_x', 'center_y', 'x1', 'y1', 'x2', 'y2']
required_targets = ['accident_time', 'center_x', 'center_y', 'type']

if not labels_clean.empty:
    for col in coord_cols:
        if col in labels_clean.columns:
            oob = ((labels_clean[col] < 0) | (labels_clean[col] > 1)).sum()
            labels_clean[col] = labels_clean[col].clip(0.0, 1.0)
            if oob > 0:
                print(f'[STATUS] Clamped {oob} out-of-bounds values in column: {col}')

    missing_mask = labels_clean[required_targets].isnull().any(axis=1)
    labels_clean = labels_clean[~missing_mask].reset_index(drop=True)
    print(f'[STATUS] Rows dropped due to missing targets: {missing_mask.sum()}')

    if 'accident_time' in labels_clean.columns: 
        labels_clean = labels_clean[labels_clean['accident_time'] >= 0].reset_index(drop=True)

display(pd.DataFrame({
    'stage'   : ['pre_clean', 'post_clean'],
    'rows'    : [pre_clean_rows, len(labels_clean)],
    'removed' : [0, pre_clean_rows - len(labels_clean)],
}))


# --- Code Cell ---
# [CLEAN] Cast frame index, resolve absolute video paths, encode type labels
if not labels_clean.empty:
    if 'accident_frame' in labels_clean.columns:
        labels_clean['accident_frame'] = labels_clean['accident_frame'].astype(int)

    if 'rgb_path' in labels_clean.columns:
        labels_clean['abs_video_path'] = labels_clean['rgb_path'].map(
            lambda p: str(resolve_video_path(p))
        )
        labels_clean['video_exists'] = labels_clean['abs_video_path'].map(
            lambda p: pathlib.Path(p).exists()
        )
        n_missing = (~labels_clean['video_exists']).sum()
        print(f'[STATUS] Records with unresolvable video paths: {n_missing}')
        labels_clean = labels_clean[labels_clean['video_exists']].reset_index(drop=True)
        display(labels_clean[['rgb_path', 'abs_video_path']].head(3))

    if 'type' in labels_clean.columns:
        TYPE_VOCAB  = sorted(labels_clean['type'].unique())
        TYPE_TO_IDX = {t: i for i, t in enumerate(TYPE_VOCAB)}
        IDX_TO_TYPE = {i: t for t, i in TYPE_TO_IDX.items()}
        labels_clean['type_idx'] = labels_clean['type'].map(TYPE_TO_IDX)
        display(pd.DataFrame({'type': TYPE_VOCAB, 'idx': range(len(TYPE_VOCAB))}))

    print('[SUCCESS] Cleaning complete')


# ===========================================================================
# ---
# ## 4. Exploratory Data Analysis
# 
# Distributions of all three prediction targets (temporal, spatial, type), scene-level tag distributions in the real test metadata, and correlation structure among numeric features.
# ===========================================================================


# --- Code Cell ---
# [EDA] Collision type frequency
if not labels_clean.empty and 'type' in labels_clean.columns:
    type_freq = labels_clean['type'].value_counts().reset_index()
    type_freq.columns = ['collision_type', 'count']

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=type_freq, x='collision_type', y='count',
                palette=list(PALETTE.values())[:len(type_freq)], ax=ax)
    ax.set_title('Collision Type Frequency (Synthetic Dataset)')
    ax.set_xlabel('Collision Type')
    ax.set_ylabel('Count')
    ax.bar_label(ax.containers[0], fontsize=9)
    plt.tight_layout()
    plt.show()


# --- Code Cell ---
# [EDA] Accident time distribution -- histogram with KDE, and box plot per type
if not labels_clean.empty and 'accident_time' in labels_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(labels_clean['accident_time'], bins=30, kde=True,
                 color=PALETTE['primary'], ax=axes[0])
    axes[0].set_title('Accident Time Distribution (seconds)')
    axes[0].set_xlabel('Accident Time (s)')
    axes[0].set_ylabel('Count')

    if 'type' in labels_clean.columns:
        sns.boxplot(data=labels_clean, x='type', y='accident_time',
                    palette='Set2', ax=axes[1])
        axes[1].set_title('Accident Time by Collision Type')
        axes[1].set_xlabel('Collision Type')
        axes[1].set_ylabel('Accident Time (s)')
        axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()


# --- Code Cell ---
# [EDA] Spatial distribution of impact points -- 2D scatter and KDE
if not labels_clean.empty and all(c in labels_clean.columns for c in ['center_x', 'center_y']):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for t_idx, (t_name, group) in enumerate(labels_clean.groupby('type')):
        axes[0].scatter(group['center_x'], group['center_y'],
                        label=t_name, alpha=0.5, s=20,
                        color=list(PALETTE.values())[t_idx % len(PALETTE)])
    axes[0].legend(fontsize=8)
    axes[0].set_title('Impact Point Distribution (Normalized Frame Coordinates)')
    axes[0].set_xlabel('center_x (0=left, 1=right)')
    axes[0].set_ylabel('center_y (0=top, 1=bottom)')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].invert_yaxis()  # match image coordinate convention

    sns.kdeplot(data=labels_clean, x='center_x', y='center_y',
                fill=True, cmap='Blues', ax=axes[1])
    axes[1].set_title('Impact Point Density (KDE)')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()


# --- Code Cell ---
# [EDA] Bounding box size distributions
if not labels_clean.empty and all(c in labels_clean.columns for c in ['x1', 'y1', 'x2', 'y2']):
    labels_clean['bbox_w']    = labels_clean['x2'] - labels_clean['x1']
    labels_clean['bbox_h']    = labels_clean['y2'] - labels_clean['y1']
    labels_clean['bbox_area'] = labels_clean['bbox_w'] * labels_clean['bbox_h']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in zip(
        axes,
        ['bbox_w', 'bbox_h', 'bbox_area'],
        ['Bounding Box Width', 'Bounding Box Height', 'Bounding Box Area'],
    ):
        sns.histplot(labels_clean[col], bins=30, kde=True, color=PALETTE['secondary'], ax=ax)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()


# --- Code Cell ---
# [EDA] Accident time as fraction of clip duration -- relative temporal position
if not labels_clean.empty and all(c in labels_clean.columns for c in ['accident_time', 'duration']):
    labels_clean['time_fraction'] = labels_clean['accident_time'] / labels_clean['duration']

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(labels_clean['time_fraction'], bins=40, kde=True,
                 color=PALETTE['tertiary'], ax=ax)
    ax.set_title('Accident Time as Fraction of Clip Duration')
    ax.set_xlabel('accident_time / duration (0=clip start, 1=clip end)')
    ax.set_ylabel('Count')
    ax.axvline(0.5, color=PALETTE['quaternary'], linestyle='--', label='midpoint')
    ax.legend()
    plt.tight_layout()
    plt.show()


# --- Code Cell ---
# [EDA] Scene condition distributions from real test metadata
SCENE_TAG_COLS = ['lighting', 'weather', 'layout']  # adjust to actual column names

if not test_df.empty:
    present_tags = [c for c in SCENE_TAG_COLS if c in test_df.columns]
    if present_tags:
        fig, axes = plt.subplots(1, len(present_tags), figsize=(6 * len(present_tags), 5))
        if len(present_tags) == 1:
            axes = [axes]
        for ax, col in zip(axes, present_tags):
            tag_freq = test_df[col].value_counts().reset_index()
            tag_freq.columns = [col, 'count']
            sns.barplot(data=tag_freq, x=col, y='count', palette='Set2', ax=ax)
            ax.set_title(f'Test Set: {col.title()} Distribution')
            ax.set_xlabel(col.title())
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        plt.show()
    else:
        print('[STATUS] Scene tag columns not found -- column names may differ')
else:
    print('[STATUS] test_df empty -- skipping scene tag plots')


# --- Code Cell ---
# [EDA] Correlation matrix across numeric features
numeric_cols = ['accident_time', 'accident_frame', 'center_x', 'center_y',
                'x1', 'y1', 'x2', 'y2', 'duration', 'no_frames',
                'height', 'width', 'bbox_w', 'bbox_h', 'bbox_area', 'time_fraction']

if not labels_clean.empty:
    present_numeric = [c for c in numeric_cols if c in labels_clean.columns]
    corr_matrix = labels_clean[present_numeric].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix: Numeric Features (Synthetic Dataset)')
    plt.tight_layout()
    plt.show()


# --- Code Cell ---
# [EDA] Sample frame extraction from a synthetic video for visual inspection
def sample_frames(video_path: pathlib.Path, n_frames: int = 6) -> list:
    """Extract n_frames evenly spaced RGB frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, total - 1, n_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

if synthetic_videos:
    sampled = sample_frames(synthetic_videos[0], n_frames=6)
    if sampled:
        fig, axes = plt.subplots(1, len(sampled), figsize=(18, 4))
        for i, (ax, frame) in enumerate(zip(axes, sampled)):
            ax.imshow(frame)
            ax.set_title(f'Frame Sample {i+1}', fontsize=10)
            ax.axis('off')
        fig.suptitle(f'Sampled Frames: {synthetic_videos[0].name}')
        plt.tight_layout()
        plt.show()
else:
    print('[STATUS] No synthetic videos found for frame sampling')


# ===========================================================================
# ---
# ## 5. Feature Engineering
# 
# Per-video feature extractors used by the inference pipeline:
# 
# 1. **Frame-difference time series** -- sharp intensity changes indicating a collision (temporal anchor).
# 2. **Optical-flow magnitude map** -- cumulative Farneback flow around the detected accident time (spatial signal and fallback).
# 3. **Frame window extraction** -- shared utility that pulls RGB frames from an arbitrary time window for the vision-language models.
# ===========================================================================


# --- Code Cell ---
# [FEATURE] Frame-difference series -- captures sharp intensity changes indicating collision

def compute_frame_diff_series(video_path: pathlib.Path,
                              resize_w: int = 320,
                              resize_h: int = 180) -> np.ndarray:
    """Mean absolute frame difference per consecutive pair.

    Returns a 1D array of length (n_frames - 1).
    """
    cap = cv2.VideoCapture(str(video_path))
    diffs, prev_gray = [], None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (resize_w, resize_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            diffs.append(np.mean(np.abs(gray - prev_gray)))
        prev_gray = gray

    cap.release()
    return np.array(diffs, dtype=np.float32)


def score_temporal_anomaly(diff_series: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    """Rolling-mean smoothing followed by z-score normalization."""
    smoothed = (pd.Series(diff_series)
                .rolling(window=smooth_window, min_periods=1, center=True)
                .mean().values)
    return (smoothed - smoothed.mean()) / (smoothed.std() + 1e-8)


print('[STATUS] compute_frame_diff_series / score_temporal_anomaly defined')


# --- Code Cell ---
# [FEATURE] Optical-flow magnitude map -- spatial localization signal

def compute_flow_magnitude_map(video_path: pathlib.Path,
                               resize_w: int = 320,
                               resize_h: int = 180,
                               n_frames_context: int = 30,
                               center_frame: int = None,
                               flow_percentile: float = 90.0) -> np.ndarray:
    """Cumulative Farneback optical-flow magnitude over a window of frames
    centered on center_frame, with percentile thresholding to suppress
    background motion. Returns a 2D array of shape (resize_h, resize_w).
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if center_frame is not None:
        start_frame = max(0, center_frame - n_frames_context // 2)
    else:
        start_frame = max(0, total // 3)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    mag_accum = np.zeros((resize_h, resize_w), dtype=np.float32)
    prev_gray, count = None, 0

    while count < n_frames_context:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (resize_w, resize_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag_accum += mag
        prev_gray = gray
        count += 1

    cap.release()

    if mag_accum.max() > 0:
        thresh = np.percentile(mag_accum, flow_percentile)
        mag_accum[mag_accum < thresh] = 0.0

    return mag_accum


print('[STATUS] compute_flow_magnitude_map defined')


# --- Code Cell ---
# [FEATURE] Frame window extraction -- shared by all vision-language stages

def extract_frames_window(video_path: pathlib.Path,
                          t_center: float,
                          t_before: float = 1.0,
                          t_after: float = 1.0,
                          n_frames: int = 8,
                          max_side: int = None) -> list:
    """Extract n_frames RGB PIL images evenly spaced over
    [t_center - t_before, t_center + t_after] seconds.

    An asymmetric window (t_before > t_after) lets classification models see
    the approach trajectories that disambiguate collision types, not just the
    post-impact wreckage. max_side optionally downscales frames to control
    VLM vision-token count.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total == 0:
        cap.release()
        return []

    start_idx = max(0, int((t_center - t_before) * fps))
    end_idx   = min(total - 1, int((t_center + t_after) * fps))
    frame_idxs = sorted(set(np.linspace(start_idx, end_idx, n_frames, dtype=int)))

    pil_frames = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_side is not None:
            w, h = img.size
            scale = max_side / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)))
        pil_frames.append(img)

    cap.release()
    return pil_frames


print('[STATUS] extract_frames_window defined')


# --- Code Cell ---
# [FEATURE] Visualize frame difference and anomaly score for a synthetic sample
if synthetic_videos:
    sample_path = synthetic_videos[0]
    diff_series = compute_frame_diff_series(sample_path)
    anomaly     = score_temporal_anomaly(diff_series)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(diff_series, color=PALETTE['primary'], linewidth=0.8)
    axes[0].fill_between(range(len(diff_series)), 0, diff_series,
                         alpha=0.2, color=PALETTE['primary'])
    axes[0].set_title('Raw Frame Difference Series')
    axes[0].set_ylabel('Mean Absolute Difference')

    axes[1].plot(anomaly, color=PALETTE['quaternary'], linewidth=0.8)
    axes[1].axhline(2.0, linestyle='--', color='grey', linewidth=0.8, label='z=2 threshold')
    axes[1].set_title('Temporal Anomaly Score (Z-Score)')
    axes[1].set_xlabel('Frame Index')
    axes[1].set_ylabel('Anomaly Z-Score')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
else:
    print('[STATUS] No synthetic videos -- skipping frame diff visualization')


# --- Code Cell ---
# [FEATURE] Visualize optical-flow magnitude map for a synthetic sample
if synthetic_videos:
    flow_map = compute_flow_magnitude_map(synthetic_videos[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(flow_map, cmap='inferno', aspect='auto')
    plt.colorbar(im, ax=ax, label='Cumulative Optical Flow Magnitude')
    ax.set_title('Optical Flow Magnitude Map (Spatial Localization Signal)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.tight_layout()
    plt.show()
else:
    print('[STATUS] No synthetic videos -- skipping flow map visualization')


# --- Code Cell ---
# [FEATURE] CLIP text prompt templates for each collision type (zero-shot scoring)
COLLISION_PROMPTS = {
    'rear-end': [
        'a car colliding into the back of another car',
        'rear-end collision between two vehicles on a road',
        'vehicle hitting the back of a stationary car from behind',
        'one car rear-ending another car at a traffic light',
        'a vehicle crashing into the tail of the car ahead',
    ],
    't-bone': [
        'a car hitting the side of another car at an intersection',
        't-bone collision at a crossroads between two vehicles',
        'side impact crash where one car strikes another perpendicularly',
        'a vehicle running a red light and hitting the side of crossing traffic',
        'perpendicular collision between two cars at a junction',
    ],
    'head-on': [
        'two cars colliding head-on from opposite directions',
        'frontal collision between two vehicles on a road',
        'head-on crash between two cars driving toward each other',
        'two vehicles smashing front-to-front on a highway',
        'a car crossing the center line and hitting an oncoming vehicle head-on',
    ],
    'sideswipe': [
        'two vehicles scraping alongside each other while driving',
        'sideswipe collision between cars changing lanes',
        'glancing blow between two cars moving in the same direction',
        'a car drifting into the adjacent lane and scraping another vehicle',
        'two vehicles brushing sides while traveling parallel on a road',
    ],
    'single': [
        'a single car crashing into a wall or barrier',
        'one vehicle running off the road and hitting an obstacle',
        'a car losing control and crashing into a pole or guardrail',
        'a single vehicle spinning out and hitting a roadside object',
        'one car veering off the road and crashing without involving another vehicle',
    ],
}

display(pd.DataFrame([
    {'collision_type': k, 'n_prompts': len(v)} for k, v in COLLISION_PROMPTS.items()
]))
print('[STATUS] Collision prompt templates defined')


# ===========================================================================
# ---
# ## 6. Modeling
# 
# The pipeline is built around one structural insight: **an accident is an event between two specific objects**, so global-frame signals (whole-frame differencing, whole-frame optical flow, whole-frame CLIP) are fundamentally noise-limited. The primary signal is therefore **per-vehicle detection and tracking**, and all three questions are answered from the kinematics of the colliding pair:
# 
# | Stage | Question | Primary signal (tracking) | Fallback (zero-shot models) |
# |-------|----------|---------------------------|------------------------------|
# | 1 -- *When* | accident time | first box contact + joint deceleration peak | frame-diff anchor + bounded PE refinement |
# | 2 -- *What* | collision type | angle between pre-impact track velocities + contact geometry | CLIP + Qwen2.5-VL soft ensemble |
# | 3 -- *Where* | impact point | centroid of the box-intersection region at impact | OWLv2 grounding + flow centroid |
# 
# Two design rules carried over from the failed experiments documented in Section 7b:
# 
# 1. **Geometry is only trusted when its preconditions hold** (long, confident, fast-moving tracks). An earlier angle classifier computed on raw optical flow failed precisely because the flow field mixed background traffic and compression noise into the velocity estimate; track velocities come from the two involved objects only.
# 2. **No hard overrides.** The geometric type hint enters a soft ensemble with CLIP and Qwen; the tracking impact time is cross-checked against the frame-difference anchor and discarded entirely when the two disagree by more than a gate (a distant "contact" is usually a false pair from occlusion, not a crash).
# ===========================================================================


# ===========================================================================
# ### 6.1 Model Loading
# 
# Two models load by default -- YOLOv8s (~0.5 GB, tracking) and CLIP ViT-B/32 (~0.6 GB, baseline classification fallback). `LOAD_LEGACY_MODELS = False` (Section 6.1 switches) keeps the Perception Encoder, Qwen2.5-VL-3B, and OWLv2 off by default -- Section 7b measured all three losing to a constant predictor, and the code that called them has been removed from the notebook (previously ~4.9 GB loaded for signals nothing used). Their measured numbers are quoted in Section 6.1's model-switch cell and in Sections 6.4 / 6.6 below, kept as documented negative results.
# 
# ===========================================================================


# --- Code Cell ---
# [MODEL] Model loading switches
#
# The VRAM audit measured 5.40 GB held by the five original models before
# Qwen3-VL-8B even started loading -- which left ~3.8 GB of a 16 GB T4 for
# activations and produced an OOM on the first clip. Qwen3-VL itself is only
# ~6.8 GB in 4-bit; the rest was models nothing calls any more.
#
# Section 7b measured each of them against its constant: the Perception Encoder
# (T=0.31 vs 0.38), OWLv2/optical-flow (S=0.15 vs 0.22), and Qwen2.5-VL-3B
# (C=0.15, below the 0.20 chance level) all lost. The loading code and the
# functions that called them (Stage 1b, 2b, 2c-ensemble, 3b) have been removed
# from the notebook -- see the correction notes in Sections 6.4 and 6.6. Only
# their measured numbers are kept, here and in those two sections.
LOAD_YOLO = True             # still used by Section 7's comparison, ~0.5 GB
LOAD_CLIP = True             # still used by Section 7's comparison, ~0.6 GB

# [MODEL] YOLOv8 -- fast per-frame vehicle detector (primary tracking signal)
YOLO_AVAILABLE = True
try:
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8s.pt')   # COCO-pretrained; small = good speed/accuracy balance
    print('[SUCCESS] YOLOv8s loaded')
except Exception as e:
    YOLO_AVAILABLE = False
    print(f'[ERROR] YOLOv8 unavailable: {e}')

# COCO class ids for vehicles: car, motorcycle, bus, truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


# --- Code Cell ---
# [MODEL] CLIP ViT-B/32 -- zero-shot classification backbone
try:
    import clip
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=DEVICE)
    clip_model.eval()
    CLIP_AVAILABLE = True
except Exception as e:
    CLIP_AVAILABLE = False
    print(f'[ERROR] CLIP unavailable: {e}')

if CLIP_AVAILABLE:
    # Pre-encode all collision-type text prompts once.
    # Following the CLIP paper's prompt-ensembling recipe, the per-type mean
    # embedding is RE-NORMALIZED after averaging. Without this, types whose
    # prompts happen to cluster tightly get a systematically larger-norm mean
    # vector and therefore inflated similarity scores.
    TYPE_TEXT_FEATURES = {}
    with torch.no_grad():
        for ctype, prompts in COLLISION_PROMPTS.items():
            tokens   = clip.tokenize(prompts).to(DEVICE)
            features = clip_model.encode_text(tokens)                    # (n_prompts, 512)
            features = features / features.norm(dim=-1, keepdim=True)
            mean_feat = features.mean(dim=0)
            TYPE_TEXT_FEATURES[ctype] = mean_feat / mean_feat.norm()
    print(f'[SUCCESS] CLIP ViT-B/32 loaded | text features for {len(TYPE_TEXT_FEATURES)} types pre-computed')


# --- Code Cell ---
# [MODEL] VRAM audit -- the two loaded models (YOLOv8s + CLIP) must fit a single T4 (16 GB)
if torch.cuda.is_available():
    print(f'[STATUS] GPU: {torch.cuda.get_device_name(0)}')
    print(f'[STATUS] VRAM allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB | '
          f'reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB | '
          f'total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB')
else:
    print('[STATUS] No CUDA device -- everything will run on CPU (very slow)')


# ===========================================================================
# ### 6.2 Vehicle Detection and Tracking (SORT-lite)
# 
# Detections are associated frame-to-frame with a minimal SORT-style tracker. The mathematical components:
# 
# **Association (Hungarian assignment).** Given predicted track boxes $\hat{B}_i$ and detections $D_j$, solve the linear assignment problem
# 
# $$\min_{x_{ij}} \sum_{i,j} c_{ij}\, x_{ij}, \qquad c_{ij} = 1 - \mathrm{IoU}(\hat{B}_i, D_j), \qquad \mathrm{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|}$$
# 
# via the Kuhn-Munkres algorithm (`scipy.optimize.linear_sum_assignment`), rejecting matches below an IoU gate.
# 
# **Motion prediction (constant velocity).** Each track's next box is its last box translated by the mean of its recent center displacements -- a zeroth-order Kalman surrogate that is adequate at 10 Hz sampling.
# 
# **Velocity estimation (central differences on smoothed centers).** After moving-average smoothing of the center sequence $c_i$, per-sample velocity uses the nonuniform central difference
# 
# $$v_i = \frac{c_{i+1} - c_{i-1}}{t_{i+1} - t_{i-1}}$$
# 
# (`np.gradient`), which is second-order accurate and does not phase-shift the estimate the way a forward difference does.
# ===========================================================================


# --- Code Cell ---
# [TRACK] IoU, box gap, and a SORT-style tracker with two-stage association
from scipy.optimize import linear_sum_assignment


def iou_xyxy(a, b) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def box_gap(a, b) -> float:
    """Euclidean separation between two boxes (0 when they touch or overlap)."""
    gx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    gy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return float(np.hypot(gx, gy))


class SortLiteTracker:
    """Minimal SORT-style multi-object tracker with ByteTrack-style association.

    Constant-velocity prediction on box centers + Hungarian assignment with an
    IoU gate. CCTV cameras are static and vehicles move smoothly, so a full
    Kalman filter adds little at 10 Hz sampling.

    Association runs in two passes over confidence-split detections. This is the
    one idea from ByteTrack that matters for this task: at the moment of impact
    the vehicles occlude and deform, detector confidence collapses, and a
    single-pass tracker with a hard confidence floor drops both tracks at
    exactly the frame the whole pipeline is trying to measure. The second pass
    sustains existing tracks from low-confidence boxes; those boxes are never
    allowed to spawn new tracks, so the noise does not leak in.
    """

    def __init__(self, iou_gate: float = 0.30, iou_gate_low: float = 0.20,
                 max_missed: int = 10, conf_high: float = 0.50):
        self.iou_gate     = iou_gate
        self.iou_gate_low = iou_gate_low
        self.max_missed   = max_missed   # 1.0 s at 10 Hz: a crash occludes for longer
        self.conf_high    = conf_high    # than the old 0.5 s, which split tracks in two
        self.tracks       = {}   # id -> {'boxes': [], 'frames': [], 'confs': [], 'missed': int}
        self.finished     = {}
        self._next_id     = 0

    def _predict(self, tr):
        """Last box translated by the mean of the last <=3 center displacements."""
        boxes = tr['boxes']
        if len(boxes) < 2:
            return boxes[-1]
        centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes[-4:]])
        d = np.diff(centers, axis=0).mean(axis=0)
        b = boxes[-1]
        return [b[0] + d[0], b[1] + d[1], b[2] + d[0], b[3] + d[1]]

    def _match(self, tids, detections, gate):
        """Hungarian match of track ids to detections above an IoU gate.

        Returns (pairs, unmatched_track_ids, unmatched_det_indices).
        """
        if not tids or not detections:
            return [], set(tids), set(range(len(detections)))

        cost = np.ones((len(tids), len(detections)))
        for i, tid in enumerate(tids):
            pred = self._predict(self.tracks[tid])
            for j, (box, _) in enumerate(detections):
                cost[i, j] = 1.0 - iou_xyxy(pred, box)

        rows, cols = linear_sum_assignment(cost)
        pairs, matched_t, matched_d = [], set(), set()
        for i, j in zip(rows, cols):
            if 1.0 - cost[i, j] >= gate:
                pairs.append((tids[i], j))
                matched_t.add(tids[i])
                matched_d.add(j)
        return pairs, set(tids) - matched_t, set(range(len(detections))) - matched_d

    def _append(self, tid, det, frame_idx):
        box, conf = det
        tr = self.tracks[tid]
        tr['boxes'].append(list(box))
        tr['frames'].append(frame_idx)
        tr['confs'].append(conf)
        tr['missed'] = 0

    def update(self, detections, frame_idx):
        """detections: list of ([x1,y1,x2,y2], conf) for one sampled frame."""
        high = [d for d in detections if d[1] >= self.conf_high]
        low  = [d for d in detections if d[1] <  self.conf_high]

        # Pass 1: confident detections, strict gate.
        pairs_h, unmatched_t, unmatched_h = self._match(
            list(self.tracks.keys()), high, self.iou_gate)
        for tid, j in pairs_h:
            self._append(tid, high[j], frame_idx)

        # Pass 2: whatever is left gets a shot at the low-confidence boxes.
        pairs_l, unmatched_t, _ = self._match(
            sorted(unmatched_t), low, self.iou_gate_low)
        for tid, j in pairs_l:
            self._append(tid, low[j], frame_idx)

        # Age out tracks that matched nothing in either pass.
        for tid in sorted(unmatched_t):
            self.tracks[tid]['missed'] += 1
            if self.tracks[tid]['missed'] > self.max_missed:
                self.finished[tid] = self.tracks.pop(tid)

        # New tracks spawn from confident detections only.
        for j in sorted(unmatched_h):
            box, conf = high[j]
            self.tracks[self._next_id] = {
                'boxes': [list(box)], 'frames': [frame_idx],
                'confs': [conf], 'missed': 0,
            }
            self._next_id += 1

    def all_tracks(self, min_len: int = 5) -> dict:
        """All tracks (finished + live) with at least min_len observations."""
        out = {}
        for tid, tr in {**self.finished, **self.tracks}.items():
            if len(tr['frames']) >= min_len:
                out[tid] = {
                    'boxes' : np.array(tr['boxes'], dtype=np.float32),
                    'frames': np.array(tr['frames'], dtype=np.int64),
                    'confs' : np.array(tr['confs'], dtype=np.float32),
                }
        return out


print('[STATUS] SortLiteTracker defined (two-stage association)')


# --- Code Cell ---
# [TRACK] Run detection + tracking over a video, then compute per-track kinematics

from scipy.signal import savgol_filter

# Two vehicles at the moment of impact overlap heavily -- often past the default
# NMS threshold of 0.7, which then deletes one of them as a duplicate box. That
# removes the collision pair from the exact frame the pipeline exists to find.
# 0.9 keeps both; the cost is a few extra duplicate boxes elsewhere, which the
# tracker's IoU gate absorbs.
YOLO_NMS_IOU  = 0.9
YOLO_CONF_MIN = 0.10   # floor for entering the tracker at all; the tracker's
                       # own conf_high splits high/low internally


def extract_vehicle_tracks(video_path: pathlib.Path,
                           sample_fps: float = 10.0,
                           conf_thresh: float = YOLO_CONF_MIN,
                           min_track_len: int = 5):
    """Detect vehicles on frames sampled at sample_fps and link them into tracks.

    Returns (tracks, fps, (W, H)) where tracks maps id -> arrays of boxes
    (pixels), frame indices, and detection confidences.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or n_frames == 0:
        cap.release()
        return {}, fps, (W, H)

    step = max(1, int(round(fps / sample_fps)))
    tracker = SortLiteTracker()

    # Sequential decode (no seeking) -- much faster than per-frame cap.set
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            results = yolo_model(frame, iou=YOLO_NMS_IOU, conf=conf_thresh,
                                 verbose=False)[0]
            detections = []
            for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                      results.boxes.cls.cpu().numpy(),
                                      results.boxes.conf.cpu().numpy()):
                if int(cls) in VEHICLE_CLASS_IDS:
                    detections.append((box.tolist(), float(conf)))
            tracker.update(detections, frame_idx)
        frame_idx += 1

    cap.release()
    return tracker.all_tracks(min_len=min_track_len), fps, (W, H)


def track_kinematics(track: dict, fps: float, smooth_window: int = 5) -> dict:
    """Smoothed center trajectory and central-difference velocity for one track."""
    boxes = track['boxes']
    t  = track['frames'] / fps
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2

    if len(cx) >= smooth_window:
        # Savitzky-Golay, NOT np.convolve(mode='same') and not a rolling mean.
        # Both distort the ends of the trajectory: convolve zero-pads, dragging
        # the first and last samples hundreds of pixels toward the origin, and a
        # centred rolling mean shrinks them by averaging over a truncated window.
        # Tracks typically END at the collision (the vehicle stops, the tracker
        # loses it) and _impact_time searches for the peak joint deceleration --
        # so an edge artifact lands precisely on the signal being measured and
        # fabricates an impact. savgol with mode='interp' fits a polynomial to
        # the edge samples instead of inventing data, and reproduces a constant
        # velocity exactly (measured: 0.0 px/s error, against 50 for a rolling
        # mean and 680 for convolve).
        #
        # savgol assumes uniform spacing; a track with missed frames is not
        # strictly uniform. The velocity below therefore still uses np.gradient
        # against the real timestamps, and the residual error at a gap is
        # unbiased rather than systematic.
        cx = savgol_filter(cx, smooth_window, 2, mode='interp')
        cy = savgol_filter(cy, smooth_window, 2, mode='interp')

    vx = np.gradient(cx, t)
    vy = np.gradient(cy, t)
    return {
        't': t, 'cx': cx, 'cy': cy, 'vx': vx, 'vy': vy,
        'speed': np.hypot(vx, vy),
        'frames': track['frames'], 'boxes': boxes,
        'conf': float(np.median(track['confs'])),
    }


print('[STATUS] extract_vehicle_tracks / track_kinematics defined')


# ===========================================================================
# ### 6.3 Collision Analysis from Track Kinematics
# 
# **Contact detection.** For each pair of concurrently tracked vehicles, the box gap $g(t)$ is monitored on their shared time support; contact is the first sample where $g(t)$ falls below 1% of the frame diagonal.
# 
# **Impact time refinement.** Within $\pm0.6$ s of contact, the impact instant is the peak of joint deceleration
# 
# $$t^{*} = \arg\max_{t} \left[ -\frac{d}{dt}\left( \lVert v_A(t) \rVert + \lVert v_B(t) \rVert \right) \right]$$
# 
# -- physically, the moment momentum is exchanged.
# 
# **Impact point.** The centroid of the intersection of the two boxes at $t^{*}$ (boxes dilated slightly when they only touch), normalized by frame size.
# 
# **Type from geometry.** With pre-impact mean velocities $\bar{v}_A, \bar{v}_B$ over $[t^{*}-1.2, t^{*}-0.2]$ s, the approach angle is
# 
# $$\theta = \arccos\left( \frac{\bar{v}_A \cdot \bar{v}_B}{\lVert \bar{v}_A \rVert\, \lVert \bar{v}_B \rVert} \right)$$
# 
# and the decision uses $\theta$ plus the contact bearing (is the struck point ahead of, beside, or behind each vehicle relative to its own motion):
# 
# | Condition | Type |
# |-----------|------|
# | $\theta \geq 135^{\circ}$ | head-on |
# | $\theta \leq 40^{\circ}$, contact along the leader's motion axis | rear-end |
# | $\theta \leq 40^{\circ}$, contact lateral | sideswipe |
# | $60^{\circ} \leq \theta \leq 120^{\circ}$ | t-bone |
# | otherwise (ambiguous bands) | no hint -- defer to VLM ensemble |
# 
# **Single-vehicle crashes** need no partner: a track whose speed collapses by more than 60% within a short window, with no other vehicle in contact, is a `single` candidate.
# 
# **Gating.** Geometry is only computed when both tracks are long ($\geq 6$ samples), confident (median detection confidence $\geq 0.4$), and moving ($\lVert \bar{v} \rVert$ above a floor) -- exactly the preconditions whose absence sank the raw-optical-flow version of this idea.
# ===========================================================================


# --- Code Cell ---
# [TRACK] Collision-pair analysis -- When / What / Where from track kinematics

MIN_SPEED_PX_S   = 15.0   # velocity direction is meaningless below this speed floor
MIN_TRACK_CONF   = 0.40
MIN_PAIR_SAMPLES = 6


def _pair_contact(kinA: dict, kinB: dict, frame_diag: float):
    """First contact between two tracks on their shared time support.

    Returns (t_contact, idxA, idxB) or None. Contact = box gap below 1% of
    the frame diagonal.
    """
    shared, ia, ib = np.intersect1d(kinA['frames'], kinB['frames'], return_indices=True)
    if len(shared) < MIN_PAIR_SAMPLES:
        return None

    gap_thresh = 0.01 * frame_diag
    gaps = np.array([box_gap(kinA['boxes'][ia[k]], kinB['boxes'][ib[k]])
                     for k in range(len(shared))])
    for k in range(len(shared)):
        if gaps[k] > gap_thresh:
            continue
        # Below the threshold but still closing means the vehicles have not
        # touched yet (e.g. a slow lateral drift before a sideswipe) --
        # contact is declared at the closest approach, not at first proximity.
        # Once boxes actually touch the gap sits at 0 and cannot keep
        # decreasing, so this always terminates at or before the true contact.
        still_approaching = (k + 1 < len(shared)
                             and gaps[k + 1] < gaps[k] - 0.02 * gap_thresh)
        if not still_approaching:
            return kinA['t'][ia[k]], ia[k], ib[k]
    return None


def _impact_time(kinA: dict, kinB: dict, t_contact: float) -> float:
    """Peak joint deceleration within +/-0.6 s of contact: the moment of
    momentum exchange, sharper than the contact sample itself."""
    grid = np.union1d(kinA['t'], kinB['t'])
    grid = grid[(grid >= t_contact - 0.6) & (grid <= t_contact + 0.6)]
    if len(grid) < 3:
        return t_contact

    joint_speed = (np.interp(grid, kinA['t'], kinA['speed'])
                   + np.interp(grid, kinB['t'], kinB['speed']))
    decel = -np.gradient(joint_speed, grid)
    return float(grid[int(np.argmax(decel))])


def _impact_point(boxA, boxB, W: int, H: int) -> tuple:
    """Centroid of the box intersection at impact, normalized to [0, 1].
    Boxes are dilated by 2% of the frame diagonal when they only touch."""
    pad = 0.02 * np.hypot(W, H)
    for p in (0.0, pad):
        ix1 = max(boxA[0] - p, boxB[0] - p)
        iy1 = max(boxA[1] - p, boxB[1] - p)
        ix2 = min(boxA[2] + p, boxB[2] + p)
        iy2 = min(boxA[3] + p, boxB[3] + p)
        if ix2 > ix1 and iy2 > iy1:
            return (float(np.clip((ix1 + ix2) / 2 / W, 0, 1)),
                    float(np.clip((iy1 + iy2) / 2 / H, 0, 1)))
    # Disjoint even after dilation: midpoint of the two centers
    cxa, cya = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
    cxb, cyb = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
    return (float(np.clip((cxa + cxb) / 2 / W, 0, 1)),
            float(np.clip((cya + cyb) / 2 / H, 0, 1)))


def _pre_impact_velocity(kin: dict, t_impact: float):
    """Mean velocity over [t_impact - 1.2, t_impact - 0.2] s (None if unobserved)."""
    mask = (kin['t'] >= t_impact - 1.2) & (kin['t'] <= t_impact - 0.2)
    if mask.sum() < 2:
        return None
    return np.array([kin['vx'][mask].mean(), kin['vy'][mask].mean()])


def _type_from_geometry(vA, vB, cA, cB) -> str:
    """Collision type from approach angle + contact bearing; None when ambiguous."""
    nA, nB = np.linalg.norm(vA), np.linalg.norm(vB)
    if nA < MIN_SPEED_PX_S or nB < MIN_SPEED_PX_S:
        return None

    cos_theta = np.clip(np.dot(vA, vB) / (nA * nB), -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))

    if theta >= 135.0:
        return 'head-on'
    if theta <= 40.0:
        # Same direction: distinguish rear-end (contact along the motion axis)
        # from sideswipe (lateral contact) via the bearing of B from A
        u = (cB - cA) / (np.linalg.norm(cB - cA) + 1e-9)
        longitudinal = abs(np.dot(vA / nA, u))
        return 'rear-end' if longitudinal >= 0.7 else 'sideswipe'
    if 60.0 <= theta <= 120.0:
        return 't-bone'
    return None  # 40-60 and 120-135 degree bands: defer to the VLM ensemble


# Feature block for the supervised type classifier of Section 6.8.
#
# Every feature here is an angle, a ratio, or a speed normalized by the frame
# diagonal. Nothing is in raw pixels and nothing touches image content. That is
# deliberate: these are the quantities that survive the CARLA-to-CCTV domain
# shift. A classifier trained on CARLA *pixels* latches onto render style and
# collapses on real footage; a classifier trained on "the two velocity vectors
# met at 93 degrees and the slower one was doing 0.2 diagonals/second" is
# reading physics that renders and real cameras agree on.
FEATURE_COLS = [
    'theta_deg', 'cos_theta', 'speed_ratio', 'speed_a_norm', 'speed_b_norm',
    'rel_speed_norm', 'longitudinal', 'drop_a', 'drop_b', 'area_ratio',
    'evidence', 'is_single', 'n_tracks',
]


def _speed_at(kin, t):
    return float(np.interp(t, kin['t'], kin['speed']))


def _pair_features(kA, kB, t_imp, vA, vB, cA, cB, frame_diag, evidence, n_tracks):
    """Domain-invariant kinematics of a collision pair. NaN where undefined --
    HistGradientBoostingClassifier consumes NaN natively, so an absent signal
    stays absent instead of being imputed into a lie."""
    f = {c: np.nan for c in FEATURE_COLS}
    f['evidence']  = float(evidence)
    f['is_single'] = 0.0
    f['n_tracks']  = float(n_tracks)

    a_pre,  b_pre  = _speed_at(kA, t_imp - 0.6), _speed_at(kB, t_imp - 0.6)
    a_post, b_post = _speed_at(kA, t_imp + 0.6), _speed_at(kB, t_imp + 0.6)
    f['speed_a_norm'] = a_pre / frame_diag
    f['speed_b_norm'] = b_pre / frame_diag
    f['speed_ratio']  = min(a_pre, b_pre) / (max(a_pre, b_pre) + 1e-9)
    f['drop_a'] = (a_pre - a_post) / (a_pre + 1e-9)
    f['drop_b'] = (b_pre - b_post) / (b_pre + 1e-9)

    areaA = float((kA['boxes'][:, 2] - kA['boxes'][:, 0]).mean()
                  * (kA['boxes'][:, 3] - kA['boxes'][:, 1]).mean())
    areaB = float((kB['boxes'][:, 2] - kB['boxes'][:, 0]).mean()
                  * (kB['boxes'][:, 3] - kB['boxes'][:, 1]).mean())
    f['area_ratio'] = min(areaA, areaB) / (max(areaA, areaB) + 1e-9)

    if vA is not None and vB is not None:
        nA, nB = np.linalg.norm(vA), np.linalg.norm(vB)
        f['rel_speed_norm'] = float(np.linalg.norm(vA - vB) / frame_diag)
        if nA >= MIN_SPEED_PX_S and nB >= MIN_SPEED_PX_S:
            cos_t = float(np.clip(np.dot(vA, vB) / (nA * nB), -1.0, 1.0))
            f['cos_theta'] = cos_t
            f['theta_deg'] = float(np.degrees(np.arccos(cos_t)))
            u = (cB - cA) / (np.linalg.norm(cB - cA) + 1e-9)
            f['longitudinal'] = float(abs(np.dot(vA / nA, u)))
    return f


def _single_features(kin, t0, frame_diag, evidence, n_tracks):
    f = {c: np.nan for c in FEATURE_COLS}
    f['evidence']  = float(evidence)
    f['is_single'] = 1.0
    f['n_tracks']  = float(n_tracks)
    pre, post = _speed_at(kin, t0 - 0.4), _speed_at(kin, t0 + 0.4)
    f['speed_a_norm'] = pre / frame_diag
    f['drop_a'] = (pre - post) / (pre + 1e-9)
    return f


def analyze_video_collision(video_path: pathlib.Path) -> dict:
    """Full tracking-based collision analysis for one video.

    Returns {'found': False, 'evidence': 0.0} when no collision is detected,
    else {'found': True, 'mode': 'pair'|'single', 't_impact': float,
     'center': (cx, cy), 'type_hint': str|None, 'evidence': float}.

    'evidence' is the normalized kinetic evidence for the contact and is the
    gate the pipeline uses to decide whether to trust this result at all.
    """
    if not YOLO_AVAILABLE:
        return {'found': False, 'evidence': 0.0, 'features': None, 'n_tracks': 0}

    tracks, fps, (W, H) = extract_vehicle_tracks(video_path)
    if not tracks or fps <= 0:
        return {'found': False, 'evidence': 0.0, 'features': None, 'n_tracks': 0}

    frame_diag = float(np.hypot(W, H))
    kins = {tid: track_kinematics(tr, fps) for tid, tr in tracks.items()}
    kins = {tid: k for tid, k in kins.items() if k['conf'] >= MIN_TRACK_CONF}

    # --- Pair collisions: pick the contact with the strongest kinetic evidence ---
    best = None
    tids = sorted(kins.keys())
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            kA, kB = kins[tids[i]], kins[tids[j]]
            contact = _pair_contact(kA, kB, frame_diag)
            if contact is None:
                continue
            t_contact, ia, ib = contact
            t_imp = _impact_time(kA, kB, t_contact)

            # Kinetic evidence = approach speed x post-contact speed drop.
            # Parked/slow pairs and drive-past occlusions score near zero.
            pre  = (np.interp(t_imp - 0.6, kA['t'], kA['speed'])
                    + np.interp(t_imp - 0.6, kB['t'], kB['speed']))
            post = (np.interp(t_imp + 0.6, kA['t'], kA['speed'])
                    + np.interp(t_imp + 0.6, kB['t'], kB['speed']))
            evidence = pre * max(0.0, pre - post)
            if pre < MIN_SPEED_PX_S:
                continue
            # Filter out passing vehicles that don't decelerate
            speed_drop_ratio = (pre - post) / (pre + 1e-9)
            if speed_drop_ratio < 0.15:
                continue
            if best is None or evidence > best['evidence']:
                best = {'kA': kA, 'kB': kB, 'ia': ia, 'ib': ib,
                        't_impact': t_imp, 'evidence': evidence}

    if best is not None:
        kA, kB = best['kA'], best['kB']
        t_imp  = best['t_impact']
        center = _impact_point(kA['boxes'][best['ia']], kB['boxes'][best['ib']], W, H)

        vA = _pre_impact_velocity(kA, t_imp)
        vB = _pre_impact_velocity(kB, t_imp)
        cA = np.array([np.interp(t_imp, kA['t'], kA['cx']),
                       np.interp(t_imp, kA['t'], kA['cy'])])
        cB = np.array([np.interp(t_imp, kB['t'], kB['cx']),
                       np.interp(t_imp, kB['t'], kB['cy'])])
        type_hint = None
        if vA is not None and vB is not None:
            type_hint = _type_from_geometry(vA, vB, cA, cB)

        # Evidence is normalized by the frame diagonal squared so the gate is
        # resolution-independent and comparable across videos. It replaces the
        # old "agrees with the frame-difference anchor" gate: that anchor scores
        # T=0.31 against a constant's 0.52, so it was validating tracking
        # against a signal weaker than a constant.
        evidence = float(best['evidence'] / (frame_diag ** 2))
        return {'found': True, 'mode': 'pair', 't_impact': round(float(t_imp), 4),
                'center': center, 'type_hint': type_hint, 'evidence': evidence,
                'n_tracks': len(kins),
                'features': _pair_features(kA, kB, t_imp, vA, vB, cA, cB,
                                           frame_diag, evidence, len(kins))}

    # --- Single-vehicle crash: speed collapse with no partner in contact ---
    best_single = None
    for tid, kin in kins.items():
        if len(kin['t']) < MIN_PAIR_SAMPLES:
            continue
        for k in range(len(kin['t'])):
            t0 = kin['t'][k]
            pre_mask  = (kin['t'] >= t0 - 0.8) & (kin['t'] < t0)
            post_mask = (kin['t'] > t0) & (kin['t'] <= t0 + 0.8)
            if pre_mask.sum() < 2 or post_mask.sum() < 2:
                continue
            pre, post = kin['speed'][pre_mask].mean(), kin['speed'][post_mask].mean()
            if pre < 2 * MIN_SPEED_PX_S:
                continue
            drop = (pre - post) / (pre + 1e-9)
            if drop >= 0.6 and (best_single is None or drop > best_single['drop']):
                box = kin['boxes'][k]
                best_single = {
                    'drop': drop, 't_impact': round(float(t0), 4), 'kin': kin, 't0': t0,
                    'evidence': float(pre * (pre - post) / (frame_diag ** 2)),
                    'center': (float(np.clip((box[0] + box[2]) / 2 / W, 0, 1)),
                               float(np.clip((box[1] + box[3]) / 2 / H, 0, 1))),
                }

    if best_single is not None:
        return {'found': True, 'mode': 'single',
                't_impact': best_single['t_impact'],
                'center': best_single['center'], 'type_hint': 'single',
                'evidence': best_single['evidence'], 'n_tracks': len(kins),
                'features': _single_features(best_single['kin'], best_single['t0'],
                                             frame_diag, best_single['evidence'],
                                             len(kins))}

    # n_tracks is reported even on failure: "tracking found nothing" and
    # "the detector saw no vehicles at all" are different diagnoses, and on
    # unlabelled real footage this is the only way to tell them apart.
    return {'found': False, 'evidence': 0.0, 'features': None,
            'n_tracks': len(kins)}


print('[STATUS] analyze_video_collision defined')


# --- Code Cell ---
# [MODEL] The constant prior -- fitted on the synthetic training set

# The real metric, from the benchmark paper via the 1st-place writeup
# (arXiv:2605.29325, Sec. 2.3). The Kaggle Evaluation page only says
# "Gaussian-style similarity"; the specification is in the papers.
#
# An earlier revision of this notebook guessed SIGMA_T = 2.0 and an isotropic
# SIGMA_S = 0.1. Both are wrong, and wrong in the flattering direction:
#
#   T is averaged over THREE tolerances, and 2.0 s is the most generous of the
#   three. Any signal with multi-second error scores near zero at 0.5 s.
#   S is ANISOTROPIC, with (sigma_x, sigma_y) set to the mean annotated bbox
#   width and height -- so vertical error is tolerated ~40% more than
#   horizontal. Independent check: the host's Molmo-7B baseline reports
#   S=0.488, which is unreachable under an isotropic 0.1.
SIGMA_T_LIST = (0.5, 1.0, 2.0)
SIGMA_X = float((labels_clean['x2'] - labels_clean['x1']).mean())
SIGMA_Y = float((labels_clean['y2'] - labels_clean['y1']).mean())
print(f'[METRIC] sigma_t = {SIGMA_T_LIST}  |  sigma_x = {SIGMA_X:.4f}  sigma_y = {SIGMA_Y:.4f}')

# Kept only so the constant grid-search below has a scalar to optimise against;
# the scoring functions in Section 7 use the real definition above.
SIGMA_T = 2.0
SIGMA_S = 0.1


def fit_constant_predictor(train_df: pd.DataFrame) -> dict:
    """Constants maximising each Gaussian component on train_df."""
    t = train_df['accident_time'].to_numpy()
    x = train_df['center_x'].to_numpy()
    y = train_df['center_y'].to_numpy()

    t_grid = np.arange(t.min(), t.max() + 1e-9, 0.05)
    t_best = float(t_grid[np.argmax([np.exp(-0.5 * ((c - t) / SIGMA_T) ** 2).mean()
                                     for c in t_grid])])

    xy_grid = np.arange(0.30, 0.71, 0.01)
    best_s, xy_best = -1.0, (0.5, 0.5)
    for cx in xy_grid:
        d2x = (cx - x) ** 2
        for cy in xy_grid:
            s = float(np.exp(-0.5 * (d2x + (cy - y) ** 2) / SIGMA_S ** 2).mean())
            if s > best_s:
                best_s, xy_best = s, (float(cx), float(cy))

    return {'accident_time': t_best,
            'center_x': xy_best[0],
            'center_y': xy_best[1],
            'type': train_df['type'].value_counts().idxmax()}


CONST = fit_constant_predictor(labels_clean)
print(f'[STATUS] Constants fitted on {len(labels_clean)} synthetic videos: '
      f"t={CONST['accident_time']:.2f}s  "
      f"xy=({CONST['center_x']:.2f}, {CONST['center_y']:.2f})  type={CONST['type']}")


# ===========================================================================
# ### 6.10 Qwen3-VL Three-Stage Pipeline
# 
# Sections 6.2–6.9 built an object-centric pipeline on the thesis that *an accident is an event between two objects, so tracking answers all three questions more directly than any whole-frame signal*. **The measurements in Sections 6.8, 6.9, and 7b refute that thesis on this data.** They are kept as a documented negative result; this section replaces them.
# 
# The 1st-place solution (arXiv:2605.29325) reaches 0.57080 with **no training at all**: a frozen Qwen3-VL checkpoint called three times per clip. Their Table 2 puts a *single* VLM call at **0.42238** — nearly double what five models and a tracker achieve here. Their reported ablations also contradict this notebook's design in three specific places:
# 
# - **"Synthetic-data transfer. A CNN classifier trained on CARLA, an optical-flow + detector hybrid, and a frame-offset ensemble all improved sim validation but decreased the LB score."** Section 6.9 is exactly this, and reproduced it: grouped-by-map accuracy **0.1464**, below the 0.20 chance level.
# - **"Overlaying object-detection tracking trails on input frames consistently degraded performance."** Tracking is not the signal.
# - Detection survives only as a **bbox snap** post-process, worth +0.0005/+0.0013.
# 
# The three stages, and why the split matters:
# 
# 1. **Stage 1 — full-video scan.** The whole clip at 4 fps with a timestamp burned into every frame, returning a joint JSON of time, location, and type. The burned-in label lets the model *copy* a timestamp instead of estimating one from visual context.
# 2. **Stage 2 — time refinement.** A dense window around the Stage-1 time, blended back as a *bounded* correction: `t = t_base + α·clip(t_refined − t_base, ±δ)` with α=0.35, δ=1.5 s. Direct replacement scored worse. **Section 6.4 of this notebook independently derived the same formula** for the Perception Encoder refinement, and measured the same failure for the unbounded version (T fell to 0.16). That instinct was right.
# 3. **Stage 3 — spatial grounding.** A single frame at the refined time, no timestamp overlay, asking for a point on Qwen3-VL's native [0,1000] scale. This is the **largest single gain in their entire development path: +0.09356**. Under a joint query the model must split attention across time and space, and the coordinates collapse onto a coarse grid.
# 
# **`scene_layout` is a provided input, not just metadata.** `test_df` carries it for every test clip (`highway`, `signalized_intersection`, `tunnel`, …) — the exact vocabulary the winner's scene-hint dictionary keys on. This notebook loaded it and labelled it *"provided for analysis, not scoring"*. It feeds two things: a hint in the Stage-1 prompt (+0.00216) and the t-bone rule (+0.00466).
# 
# **Hardware.** Their 32B-FP8 leg needs ~13 h on an RTX PRO 6000 and does not fit a T4. `VLM_CHECKPOINT` below defaults to **Qwen3-VL-8B**, which their backbone sweep puts at 0.365 single-call at 512 px against 0.387 for the 32B — most of the quality at a quarter of the size. **Runtime on a T4 is the main open risk and is not yet measured: use the `SUBSET_N` dry run before committing to a full pass.**
# ===========================================================================


# --- Code Cell ---
# [MODEL] Qwen3-VL -- the backbone that replaces the tracking pipeline
#
# The reference solution serves Qwen3-VL-32B-Instruct-FP8 through vLLM. Two
# reasons this cell does not: FP8 needs SM89+ (a T4 is SM75), and 32B does not
# fit 16 GB. transformers + 4-bit is what this environment is known to run --
# the existing Qwen2.5-VL cell already does it. vLLM would be substantially
# faster and is worth switching to if 2027 clips prove too slow.
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

VLM_CHECKPOINT = 'Qwen/Qwen3-VL-8B-Instruct'

# Reference config (32B): 4 fps, <=128 frames, 960 px. Their Table 2 measures a
# single call at 768 px / 2 fps = 0.42238, and going 768->960 px and 2->4 fps was
# worth +0.01969. The conservative setting is the default here because a T4 has
# to hold the KV cache for every frame; raise them if the dry run has headroom.
# 64 frames at 768 px OOM'd a T4 on the first clip. 512 px is not a guess: it is
# the resolution of the reference backbone sweep, where this same 8B checkpoint
# scored 0.365 single-call. Frame count is the other half of the activation
# budget; raise both only if the memory report above shows headroom.
# Frame budget. The earlier 1 fps / 24 f / 512 px was set blind, to survive an
# OOM whose real cause was 4.9 GB of dead legacy models plus device_map pinning
# everything to cuda:0. With those fixed the loader reports ~10 GB free on the
# smaller of the two T4s, so the budget can go back up -- and it matters: the
# reference measures 768 px / 2 fps at 0.42238, and their 512 px backbone sweep
# puts the 8B at 0.365. Running below their baseline configuration gives up
# score before the pipeline even starts.
#
# Raise 'whole_limit' toward 128 and 'image_max_side' toward 960 if the dry run
# shows headroom; drop them first if anything OOMs.
# A Qwen-VL "visual token" covers a 28x28 pixel block, so a 768 px frame of a
# 16:9 clip costs 423 tokens and 48 of them cost 20,304 -- which is what the OOM
# on every calibration clip was. The retry then halved the frame count twice,
# so Stage 1 answered from 12 frames spread over ~20 s: roughly one frame every
# 1.7 s, which can miss the collision entirely. That is the likely cause of both
# the wrong times and the collapsed types.
#
# The two stages want different things, so they no longer share a budget:
#   Stage 1 answers "when" and "what" -> needs TEMPORAL COVERAGE, so many frames
#           at low resolution (32 x 448 px = 4,608 tokens).
#   Stage 3 answers "where" in one frame -> needs SPATIAL DETAIL, so a single
#           frame at high resolution (768 px = 423 tokens, cheap at n=1).
# The reference reports the same asymmetry from the other side: going 4->10 fps
# cost them -0.009, and rendering the grounding frame at 1024/1280 px degraded
# spatial accuracy. More pixels is not monotonically better.
VLM_CFG = {
    'whole_fps': 2.0,
    'whole_limit': 32,
    'image_max_side': 448,
    'grounding_max_side': 768,
    'split_threshold_sec': 32.0,
    'pass_duration_sec': 32.0,
    'time_refine': True,
    'time_refine_window': 2.0,      # dense window half-width, seconds
    'time_refine_fps': 4.0,
    'time_refine_limit': 10,
    'time_refine_context_before': 8.0,
    'time_refine_context_after': 4.0,
    'time_refine_context_fps': 0.5,
    'time_refine_context_limit': 4,
    'time_refine_blend': 0.35,      # alpha -- correction is down-weighted
    'time_refine_max_shift_sec': 1.5,   # delta_max -- and hard-capped
    'use_grounding': True,
}

VLM_AVAILABLE = True
try:
    # A previous failed/partial load can leave tensors alive in this namespace,
    # which torch.cuda.empty_cache() alone will not reclaim -- it only returns
    # cache the allocator itself is not still holding a live reference to. Drop
    # any stale handle before loading again, in case this cell is re-run without
    # a kernel restart.
    for _name in ('vlm_model', 'vlm_processor'):
        if _name in globals():
            del globals()[_name]
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _bnb = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            if DEVICE == 'cuda' else None)
    vlm_processor = AutoProcessor.from_pretrained(VLM_CHECKPOINT)

    # device_map='auto' shards the model across BOTH T4s by pre-load free memory,
    # but it balances STATIC WEIGHTS only -- it has no way to know that decode-time
    # KV-cache growth (which scales with visual-token count, i.e. whole_limit x
    # image_max_side) will land on whichever GPU ends up owning the later decoder
    # layers. On this run it concentrated most of the model's ~6.8 GB *and* the
    # growing KV-cache onto GPU 1, which then had nothing left for a 938 MiB
    # allocation. The fix is not "prefer GPU 1" (that would just hardcode today's
    # imbalance and break on a single-GPU box, e.g. a laptop) -- it is to pin the
    # WHOLE model onto whichever single GPU has the most free memory right now,
    # since 6.8 GB comfortably fits alone on one 14-16 GB T4 with room to spare
    # for activations. No sharding, no cross-GPU imbalance to reason about.
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

    # Confirm bitsandbytes actually replaced the Linear layers. If the config is
    # silently ignored the model loads in fp16, takes ~16 GB, and the OOM appears
    # later and further away, looking like a frame-budget problem instead.
    _n4bit = sum(1 for m in vlm_model.modules() if 'Linear4bit' in type(m).__name__)
    print(f'[STATUS] Linear4bit layers: {_n4bit}  (0 means quantization did NOT apply)')
except Exception as e:
    VLM_AVAILABLE = False
    print(f'[ERROR] Qwen3-VL unavailable: {type(e).__name__}: {e}')

if torch.cuda.is_available():
    for _g in range(torch.cuda.device_count()):
        _free, _tot = torch.cuda.mem_get_info(_g)
        print(f'[STATUS] GPU{_g} {torch.cuda.get_device_name(_g)}: '
              f'{(_tot-_free)/1e9:.2f} GB used / {_tot/1e9:.2f} GB total, '
              f'{_free/1e9:.2f} GB free for activations')

if not VLM_AVAILABLE:
    raise RuntimeError(
        'Qwen3-VL is not usable, so Section 6.10 cannot run. Do NOT continue to the '
        'submission cell: run_inference_vlm would return normalize_prediction defaults '
        'for every clip -- a constant dressed up as a prediction. Fix the environment '
        '(see the install cell: transformers must be >=4.57 AND the kernel restarted '
        'afterwards) before going on.')


# --- Code Cell ---
# [MODEL] Prompts and frame sampling for the three stages

# scene_layout is a provided column of test_df. Keys match its vocabulary.
SCENE_TYPE_HINTS = {
    'highway': 'Common accident types here: rear-end (following too closely), single (loss of control, hitting barrier/guardrail), or sideswipe (lane change).',
    'signalized_intersection': 'Common accident types here: t-bone (running red light, perpendicular collision), rear-end (stopping suddenly), or single (running off road).',
    'simple_intersection': 'Common accident types here: t-bone (failure to yield at unsignalized intersection), rear-end, or single.',
    'grade_separated_intersection': 'Common accident types here: single (loss of control on ramp/merge), rear-end (merging traffic), or t-bone.',
    'city_street': 'Common accident types here: t-bone (intersection), rear-end, single (hitting parked car/pole), or sideswipe.',
    'tunnel': 'Common accident types here: rear-end (reduced visibility), single (hitting wall), or sideswipe.',
    'parking_lot': 'Common accident types here: single (hitting pole/wall), rear-end (backing up), or sideswipe.',
    'roundabout': 'Common accident types here: t-bone (failure to yield), sideswipe (lane change in roundabout), or single.',
}

# Two vehicles cannot meet perpendicularly in these layouts.
T_BONE_IMPOSSIBLE_LAYOUTS = {'highway', 'tunnel', 'grade_separated_intersection'}

# Scene metadata keyed by submission path, for the hint and the rule.
SCENE_BY_PATH = (test_df.set_index('path')['scene_layout'].to_dict()
                 if not test_df.empty and 'scene_layout' in test_df.columns else {})


def build_stage1_prompt(duration: float, scene_hint: str) -> str:
    return (
        'These sequential CCTV frames show a traffic accident. '
        'Each frame has a burned-in timestamp (t=xx.xx s). '
        f'The full video is {duration:.1f} seconds long.\n\n'
        'Your task: Find the EXACT MOMENT, LOCATION, and TYPE of the collision.\n\n'
        "1) COLLISION TIME -- Carefully examine each frame's timestamp. "
        'Find the frame where vehicles first make contact or a vehicle first hits an object. '
        'Report the timestamp in seconds.\n\n'
        '2) COLLISION POSITION -- Where in the frame does the impact happen? '
        'Report as normalized coordinates: center_x (0=left, 1=right), center_y (0=top, 1=bottom).\n\n'
        "3) ACCIDENT TYPE -- Classify the collision type by watching the vehicles' "
        'approach angles and movements across all frames:\n'
        '  - rear-end = a following vehicle strikes the one ahead (both traveling same direction)\n'
        '  - head-on = two vehicles collide front-to-front (traveling opposite directions)\n'
        '  - sideswipe = vehicles traveling parallel make side-to-side contact\n'
        "  - t-bone = perpendicular collision (one vehicle hits the other's side at ~90 degrees)\n"
        '  - single = only one vehicle involved (hits object, barrier, pole, or loses control)\n'
        f'{scene_hint}\n'
        'Look at:\n'
        '- How many vehicles are involved (one = single)\n'
        '- The direction each vehicle is traveling before impact\n'
        '- The angle of collision (same direction = rear-end, opposite = head-on, '
        'perpendicular = t-bone, parallel side = sideswipe)\n\n'
        'Output ONLY this JSON:\n'
        '{"accident_time": <seconds>, "center_x": <0-1>, "center_y": <0-1>, '
        '"type": "rear-end|head-on|t-bone|sideswipe|single"}'
    )


TIME_REFINE_PROMPT = (
    'These sequential CCTV frames show a traffic accident. '
    'Some frames provide broader context before/after the event, and other frames densely '
    'cover a short window around the current predicted collision time. '
    'Each frame has a burned-in timestamp (t=xx.xx s).\n\n'
    'Your task: identify the EXACT timestamp when vehicles first make physical contact, '
    'or when a vehicle first hits an object. '
    'Do NOT choose the peak impact, aftermath, debris, spin, or when vehicles are already '
    'clearly entangled. Use the broader context to understand motion, but report ONLY the '
    'first contact time.\n\n'
    'Output ONLY this JSON:\n'
    '{"accident_time": <seconds>}'
)

GROUNDING_PROMPT = (
    'This CCTV frame shows a traffic accident. '
    'Point to the EXACT location where the collision or impact occurs. '
    'Output the collision point coordinates as JSON:\n'
    '{"point": [x, y]}\n'
    'where x is horizontal (0=left edge, 1000=right edge) and '
    'y is vertical (0=top edge, 1000=bottom edge).'
)


def sample_frames_stamped(video_path, target_fps, t_start=None, t_end=None,
                          limit=64, max_side=768, burn=True):
    """Frames at target_fps in [t_start, t_end], each with its timestamp drawn on.

    The burned-in label is the point: it lets the model read a timestamp off the
    pixels and copy it into the answer, instead of inferring elapsed time from
    visual context. Returns [(t_seconds, PIL.Image), ...].
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or n <= 0:
        cap.release()
        return []
    duration = n / fps
    t0 = 0.0 if t_start is None else max(0.0, t_start)
    t1 = duration if t_end is None else min(duration, t_end)
    if t1 <= t0:
        t1 = min(duration, t0 + 1.0)

    times = np.arange(t0, t1 + 1e-6, 1.0 / max(target_fps, 0.1))
    if len(times) > limit:
        times = times[np.linspace(0, len(times) - 1, limit).round().astype(int)]

    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.clip(round(t * fps), 0, n - 1)))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            sc = max_side / float(max(h, w))
            frame = cv2.resize(frame, (int(round(w * sc)), int(round(h * sc))),
                               interpolation=cv2.INTER_AREA)
        if burn:
            cv2.rectangle(frame, (8, 8), (250, 46), (0, 0, 0), -1)
            cv2.putText(frame, f't={t:.2f}s', (14, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
        out.append((float(t), PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    return out


def _extract_json(text):
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def vlm_generate(prompt_text, images, max_new_tokens=256):
    """Greedy decode over a list of PIL frames. Greedy, not sampled: the
    reference reports self-consistency at temperature 0.1-0.6 changing fewer
    than ~10 predictions in 2027 clips, and CoT collapsing the temporal score
    from 0.42 to 0.06."""
    if not VLM_AVAILABLE:
        return ''
    content = [{'type': 'image', 'image': im} for im in images]
    content.append({'type': 'text', 'text': prompt_text})
    inputs = vlm_processor.apply_chat_template(
        [{'role': 'user', 'content': content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt',
    ).to(vlm_model.device)
    with torch.no_grad():
        out = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = vlm_processor.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    del inputs, out
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return text


print('[STATUS] prompts / sample_frames_stamped / vlm_generate defined')

def vlm_generate_oom_safe(prompt_text, frames, max_new_tokens=256, min_frames=4):
    """vlm_generate with OOM retry: halves the frame list and retries instead of
    crashing. frames is a list of (t, PIL.Image) tuples, as returned by
    sample_frames_stamped -- passing the tuples (not just the images) lets this
    halve consistently no matter which stage calls it. Returns '' if even
    min_frames still OOMs, so a caller can fall back to a default rather than die.

    Every direct vlm_generate() call in this notebook that runs inside a loop
    over many videos (Stage 2, Stage 3, the debug dump) should go through this
    instead: stage1_full_scan already had its own retry loop, but Stage 2/3 and
    the diagnostic cells did not, and a single oversized clip could kill an
    otherwise-checkpointed multi-hour run.
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
                print(f'[WARNING] OOM persists at {len(cur)} frames -- giving up, returning empty string')
                return ''
            cur = cur[::2] if len(cur) > 2 * min_frames else cur[:min_frames]
            print(f'[WARNING] OOM -- retrying with {len(cur)} frames')



# ===========================================================================
# ### Fix -- NVRTC builtins missing (`libnvrtc-builtins.so.13.0`)
# 
# Baseline torch is `2.11.0+cu130`, and the simple smoke-test op (`tensor * 2`)
# passed -- that op uses a PRECOMPILED kernel. The canary's `generate()` call hit
# a reduction op (`prod`) that has no precompiled kernel for its dtype/shape and
# falls back to JIT-compiling one via NVRTC at runtime, which needs
# `libnvrtc-builtins.so.13.0` on disk. That file is missing or mismatched --
# likely because this torch build was already sitting in the Kaggle image
# (not a normal `pip install torch`), so its CUDA 13.0 runtime sub-packages
# (`nvidia-cuda-nvrtc-cu13` etc.) never went through pip's own dependency
# resolution. Run the cell below, then re-run the canary (no kernel restart
# needed -- NVRTC is `dlopen`'d lazily on first JIT call, not at import time;
# restart only if the retry below still fails).
# ===========================================================================


# --- Code Cell ---
# [FIX] Install/repair the matching NVRTC runtime for torch 2.11.0+cu130
import subprocess, sys, glob

_torch_cuda = None
try:
    import torch
    _torch_cuda = torch.version.cuda
except Exception as e:
    print(f'[WARNING] could not import torch to read version: {e}')
print(f'[STATUS] torch reports CUDA {_torch_cuda}')

_rc, _out, _err = subprocess.run(
    [sys.executable, '-m', 'pip', 'show', 'nvidia-cuda-nvrtc-cu13'],
    capture_output=True, text=True
).returncode, '', ''
_show = subprocess.run([sys.executable, '-m', 'pip', 'show', 'nvidia-cuda-nvrtc-cu13'],
                       capture_output=True, text=True)
print(f'[STATUS] nvidia-cuda-nvrtc-cu13 currently installed: {_show.returncode == 0}')
if _show.returncode == 0:
    print(_show.stdout)

# Find whatever libnvrtc-builtins*.so actually exists on disk right now, for comparison
_found = glob.glob(sys.prefix + '/lib/python*/site-packages/nvidia/cuda_nvrtc/**/libnvrtc-builtins*.so*',
                   recursive=True)
print(f'[STATUS] libnvrtc-builtins*.so found on disk: {_found}')

print('[STATUS] Reinstalling nvidia-cuda-nvrtc-cu13 (force, matching pip resolver -- '
      'not --no-deps, so pip can pick a build compatible with installed torch)...')
_repair = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
     'nvidia-cuda-nvrtc-cu13'],
    capture_output=True, text=True
)
print(f'[STATUS] Reinstall exit code: {_repair.returncode}')
if _repair.returncode != 0:
    print(_repair.stderr[-2000:])

_found_after = glob.glob(sys.prefix + '/lib/python*/site-packages/nvidia/cuda_nvrtc/**/libnvrtc-builtins*.so*',
                         recursive=True)
print(f'[STATUS] libnvrtc-builtins*.so found after reinstall: {_found_after}')

# Retry the exact op class that failed -- a JIT reduction, not the plain smoke-test op
try:
    import torch
    _t = torch.tensor([2, 3, 4], dtype=torch.int64, device='cuda')
    _p = torch.prod(_t)
    print(f'[SUCCESS] JIT reduction op works: prod([2,3,4]) = {_p.item()}')
except Exception as e:
    print(f'[ERROR] JIT reduction still fails: {type(e).__name__}: {e}')
    print('[NEXT] If this still fails, Restart Kernel and re-run from the top -- the')
    print('       repair above only replaces the file on disk, and if any process')
    print('       already opened a handle to the old/missing one this session, a')
    print('       fresh process is the only way to pick up the reinstalled library.')



# ===========================================================================
# ### Fix v2 -- `nvidia-cuda-nvrtc-cu13` khong co wheel dung san, bo huong vay lai NVRTC
# 
# `pip install nvidia-cuda-nvrtc-cu13` that bai ngay o buoc BUILD WHEEL (khong phai
# loi mang/OOM) -- nghia la khong co prebuilt wheel cho package nay tren PyPI voi
# platform/Python hien tai (CUDA 13.0 con qua moi). Vay lai dung file .so thieu la
# ngo cut thuc su, khong phai do lam sai buoc nao.
# 
# **Doi huong**: dung dung nhanh tu-sua da co san trong cell 8 (comment da noi ro:
# "a safer bet than whatever cu128 build shipped with this image") nhung LAN NAY
# CHAY NO CHU DONG, khong doi baseline smoke-test tu bao loi -- vi da chung minh
# smoke-test hien tai (`tensor * 2`) qua don gian de bat loi JIT nay, no "pass gia".
# Cell duoi ha thang ve `cu121` (torch/torchvision/torchaudio on dinh, da chay
# sm_75/T4 lien tuc nhieu nam) va nang cap chinh smoke-test de bao gom phep JIT-
# reduction (`torch.prod`) cho lan sau khong bi lua nua.
# 
# **BAT BUOC Restart Kernel sau khi cell nay chay xong** -- torch cu130 da duoc
# import vao process nay roi, pip reinstall khong thay the duoc module dang nam
# trong bo nho; phai la process moi (kernel moi) moi nap dung ban cu121.
# ===========================================================================


# --- Code Cell ---
# [FIX v2] Ha torch ve cu121 on dinh -- KHONG doi smoke-test bao loi truoc,
# chu dong ha vi da xac nhan cu130 hong o JIT-reduction va khong sua duoc.
import subprocess, sys

print('[STATUS] Force-reinstalling torch/torchvision/torchaudio tu cu121 index...')
_rc = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
     'torch', 'torchvision', 'torchaudio',
     '--index-url', 'https://download.pytorch.org/whl/cu121'],
    capture_output=True, text=True
)
print(f'[STATUS] cu121 install exit code: {_rc.returncode}')
if _rc.returncode != 0:
    print(_rc.stderr[-2000:])
else:
    # Re-pin the constraints file used by later installs (ultralytics/CLIP/transformers)
    # to the NEW cu121 versions, so nothing pulls cu130 back in as a transitive dep.
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

print()
print('[ACTION REQUIRED] Restart Kernel now, then run from the top.')
print('  Cell 8 will re-pin these same versions (constraints file already updated),')
print('  so ultralytics / CLIP / perception_models / transformers installs will not')
print('  pull torch back to cu130.')



# --- Code Cell ---
# [CHECK] The canary -- does the VLM actually generate?
#
# Loading without raising is not the same as working. A previous run loaded
# nothing, returned '' from every call, and filled all 2027 rows with
# normalize_prediction's defaults (0.35*duration, 0.5/0.5, rear-end): a constant
# wearing a submission's clothes, produced after 24 minutes of GPU time.
#
# An earlier attempt to prevent that put this check in the loader cell, guarded
# by "if 'vlm_generate' in globals()" -- and since vlm_generate is defined below
# the loader, the guard silently skipped the check on every run. A safety check
# that can quietly not run is not a safety check. This cell has no guard.
_probe = PILImage.fromarray(np.full((448, 448, 3), 127, np.uint8))
_out = vlm_generate('Reply with exactly: OK', [_probe], max_new_tokens=8)
print(f'[CANARY] generation returned: {_out!r}')
assert _out.strip(), 'The VLM loaded but generates nothing. Do not continue.'
print('[CANARY] PASS -- the model is alive')


# --- Code Cell ---
# [MODEL] The three stages

def _clip01(v, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, v)))


def _safe_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def normalize_prediction(pred, duration):
    """Coerce a raw VLM JSON into a valid submission row.

    The default time is 0.35*duration, not 0.5: accidents sit earlier in a clip
    than the midpoint. This is a per-clip parse failure handler ONLY -- if it is
    firing on every clip, the model is not running and the caller must fail loudly
    rather than emit these defaults 2027 times. run_inference_vlm enforces that.
    """
    out = dict(pred or {})
    out['accident_time'] = _clip01(_safe_float(out.get('accident_time'), duration * 0.35),
                                   0.0, duration)
    out['center_x'] = _clip01(_safe_float(out.get('center_x'), 0.5))
    out['center_y'] = _clip01(_safe_float(out.get('center_y'), 0.5))
    t = str(out.get('type') or 'rear-end').strip().lower()
    out['type'] = t if t in COLLISION_TYPES else 'rear-end'
    return out


def stage1_full_scan(video_path, duration, scene_layout=None):
    """Joint (time, location, type) over the whole clip. Already a complete answer.

    Retries on OOM with half the frames rather than dying on one long 4K clip:
    the real test set ranges from 5.9 s to 30 s and up to 3840x2160, so the
    activation cost per clip varies by more than an order of magnitude. The
    retry prints -- a clip silently answered from 6 frames is worth knowing about.
    """
    hint = SCENE_TYPE_HINTS.get(scene_layout or '', '')
    limit = VLM_CFG['whole_limit']
    while True:
        frames = sample_frames_stamped(video_path, VLM_CFG['whole_fps'], limit=limit,
                                       max_side=VLM_CFG['image_max_side'], burn=True)
        if not frames:
            return {**normalize_prediction({}, duration), '_parsed': False}
        try:
            raw = vlm_generate(build_stage1_prompt(duration, hint), [im for _, im in frames])
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if limit <= 6:
                raise
            limit //= 2
            print(f'[WARNING] OOM on {video_path.name} -- retrying with {limit} frames')
    parsed = _extract_json(raw)
    out = normalize_prediction(parsed, duration)
    out['_parsed'] = parsed is not None
    return out


def stage2_time_refine(video_path, t_base, duration):
    """Bounded correction to t_base -- never a replacement.

    A dense window around t_base plus a few sparse context anchors outside it, so
    the model still sees the motion leading into and out of the event. The
    correction is down-weighted by alpha and hard-capped at delta_max because
    ACCS is a harmonic mean: one large error costs more than many small ones buy.
    Section 6.4 derived this same bounded form for the PE refinement and measured
    the unbounded variant regressing T to 0.16.
    """
    if not VLM_CFG['time_refine']:
        return t_base
    w = VLM_CFG['time_refine_window']
    dense = sample_frames_stamped(video_path, VLM_CFG['time_refine_fps'],
                                  t_base - w, t_base + w,
                                  limit=VLM_CFG['time_refine_limit'],
                                  max_side=VLM_CFG['image_max_side'], burn=True)
    ctx = sample_frames_stamped(video_path, VLM_CFG['time_refine_context_fps'],
                                t_base - VLM_CFG['time_refine_context_before'],
                                t_base + VLM_CFG['time_refine_context_after'],
                                limit=VLM_CFG['time_refine_context_limit'],
                                max_side=VLM_CFG['image_max_side'], burn=True)
    merged = sorted({round(t, 3): im for t, im in (ctx + dense)}.items())
    if not merged:
        return t_base
    j = _extract_json(vlm_generate_oom_safe(TIME_REFINE_PROMPT, merged, 64))
    if not j or 'accident_time' not in j:
        return t_base
    t_ref = _safe_float(j['accident_time'], t_base)
    delta = np.clip(t_ref - t_base, -VLM_CFG['time_refine_max_shift_sec'],
                    VLM_CFG['time_refine_max_shift_sec'])
    return float(np.clip(t_base + VLM_CFG['time_refine_blend'] * delta, 0.0, duration))


def stage3_grounding(video_path, t_final):
    """Point at the impact in ONE frame at t_final. No timestamp overlay here --
    the model has nothing left to read but the collision.

    Largest single gain in the reference development path (+0.09356). Under the
    Stage-1 joint query the model splits attention between when and where, and
    the coordinates quantise onto a coarse grid; with the time already pinned,
    this asks only 'where'.
    """
    if not VLM_CFG['use_grounding']:
        return None
    # One frame, so it can afford the resolution Stage 1 cannot.
    frames = sample_frames_stamped(video_path, 1.0, t_final, t_final + 1e-3, limit=1,
                                   max_side=VLM_CFG.get('grounding_max_side',
                                                        VLM_CFG['image_max_side']),
                                   burn=False)
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
    if pt is None:
        m = re.search(r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)', text)
        if m:
            pt = (float(m.group(1)), float(m.group(2)))
    if pt is None or any(np.isnan(v) for v in pt):
        return None
    x, y = pt
    if x > 1.0 or y > 1.0:      # Qwen3-VL's native grounding scale is [0, 1000]
        x, y = x / 1000.0, y / 1000.0
    return (_clip01(x), _clip01(y))


def apply_scene_type_postfix(accident_type, scene_layout):
    """t-bone is geometrically impossible where traffic cannot cross."""
    if accident_type == 't-bone' and scene_layout in T_BONE_IMPOSSIBLE_LAYOUTS:
        return 'rear-end'
    return accident_type


print('[STATUS] stage1_full_scan / stage2_time_refine / stage3_grounding defined')


# ===========================================================================
# ### Fix -- bien/ham bi xoa nham trong dot don dead-code truoc
# 
# `TYPE_DEFINITIONS`, `CLASSIFICATION_PROMPTS`, `_parse_type_label` von nam trong
# cell "Stage 2b -- Qwen2.5-VL multi-prompt voting" (da xoa vi 2 ham chinh cua no,
# `_qwen_generate` va `qwen_type_votes`, thuc su chet). Nhung 3 ten nay VAN duoc
# `classify_type_cascade` (Section 6.5, dang song) dung -- kiem tra lai lan nay
# lam dung (soat TAT CA ten dinh nghia trong 7 cell da xoa, khong chi ham, ma ca
# hang so/dict), xac nhan day la landmine DUY NHAT trong 7 cell do; 6 cell con
# lai (44,45,46,55,60,63 theo so cu) an toan, khong con gi tham chieu toi.
# ===========================================================================


# --- Code Cell ---
# [FIX] TYPE_DEFINITIONS / CLASSIFICATION_PROMPTS / _parse_type_label -- khoi
# phuc dung nguyen ban goc, dat truoc classify_type_cascade (Section 6.5) can no.
TYPE_DEFINITIONS = (
    "head-on = two vehicles traveling in opposite directions collide front-to-front; "
    "rear-end = the front of one vehicle hits the back of another traveling the same direction; "
    "sideswipe = two roughly parallel vehicles make glancing side contact; "
    "t-bone = the front of one vehicle strikes the side of another at a right angle; "
    "single = exactly one vehicle crashes (wall, pole, barrier, or off the road), no second vehicle involved."
)

CLASSIFICATION_PROMPTS = {
    'count-first': (
        "These frames from a CCTV camera show a traffic accident in chronological order. "
        "First count how many moving vehicles are directly involved in the crash. "
        "If exactly one vehicle crashes on its own, the answer is 'single'. "
        f"Otherwise classify the collision. Definitions: {TYPE_DEFINITIONS} "
        "Answer with only one label: head-on, rear-end, sideswipe, single, or t-bone."
    ),
    'motion': (
        "Watch the direction each vehicle travels across these chronological frames "
        "up to the moment of impact. Based on their directions of travel at contact, "
        f"classify the collision. Definitions: {TYPE_DEFINITIONS} "
        "Answer with only one label: head-on, rear-end, sideswipe, single, or t-bone."
    ),
    'geometry': (
        "Focus on the angle and first contact point between the vehicles at the moment "
        f"of collision in these frames. Definitions: {TYPE_DEFINITIONS} "
        "Answer with only one label: head-on, rear-end, sideswipe, single, or t-bone."
    ),
}


def _parse_type_label(text: str) -> str:
    normalized_text = text.lower().replace(' ', '-').replace('_', '-')
    for label in COLLISION_TYPES:
        if label in normalized_text:
            return label
    return None


print('[STATUS] TYPE_DEFINITIONS / CLASSIFICATION_PROMPTS / _parse_type_label restored')



# --- Code Cell ---
# [MODEL] Stage 2 (cascade) -- entropy/margin-gated classification, Qwen3-VL

from collections import Counter

# stage1_full_scan's `type` comes from ONE JSON call with no cross-check --
# the same failure diagnosed for Qwen2.5-VL in Section 7b (a model that
# settles on one class far more often than the label distribution supports).
# Section 6.5's fix for that model was an unconditional multi-prompt ensemble
# on every clip; that is not affordable here (Qwen3-VL-8B is already the
# session's VRAM ceiling -- Section 6.1's audit). Instead: escalate only when
# cheap signals disagree, per the metadata-aware cascade baseline -- most
# clips exit after two extra short calls confirm what Stage 1 already said.
ELIMINATION_PROMPT_TEMPLATE = (
    "These CCTV frames show a traffic accident. Two earlier analyses of this "
    "same clip disagreed: one concluded '{a}', the other concluded '{b}'. "
    f"Definitions: {TYPE_DEFINITIONS} "
    "Look carefully at how many vehicles are involved, the direction each one "
    "is traveling, and the angle of contact, then decide which of the two "
    "labels is correct. Answer with only one label: {a} or {b}."
)


def _vote_entropy(counts) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return float(-sum(p * np.log2(p) for p in probs))


def classify_type_cascade(video_path: pathlib.Path, t_final: float, type_from_stage1: str) -> str:
    """Entropy/margin-gated classification.

    votes = [stage1's own type (free -- already computed), 'motion' prompt,
    'geometry' prompt]. If >=2 of 3 agree and entropy stays low, that majority
    is the answer -- two extra cheap VLM calls, no more. Only on a genuine
    3-way split does this pay for one more, explicit elimination call between
    the top two candidates, rather than trusting a plurality vote on 3 items
    (which is really just "pick whichever 2 of the 3 short prompts happened
    to agree").
    """
    if not VLM_AVAILABLE:
        return type_from_stage1

    frames = sample_frames_stamped(video_path, 2.0, t_final - 1.5, t_final + 1.0,
                                   limit=6, max_side=VLM_CFG['image_max_side'], burn=False)
    if not frames:
        return type_from_stage1

    votes = [type_from_stage1]
    for prompt in CLASSIFICATION_PROMPTS.values():
        label = _parse_type_label(vlm_generate_oom_safe(prompt, frames, 10, min_frames=2))
        if label is not None:
            votes.append(label)

    counts = Counter(votes)
    top_label, top_n = counts.most_common(1)[0]

    # Full or majority (>=2 of <=3) agreement with low entropy: trust it, done.
    if top_n >= 2 and _vote_entropy(counts) <= 1.0:
        return top_label

    # Every vote disagreed (entropy at its max for 3 items) -- escalate with
    # one elimination call between the two most-recent, most-informed votes:
    # Stage 1 saw the whole clip, 'geometry' looks at contact angle directly.
    candidates = list(dict.fromkeys([type_from_stage1, votes[-1]]))
    if len(candidates) < 2:
        return top_label
    a, b = candidates[0], candidates[1]
    prompt = ELIMINATION_PROMPT_TEMPLATE.format(a=a, b=b)
    label = _parse_type_label(vlm_generate_oom_safe(prompt, frames, 10, min_frames=2))
    return label if label in (a, b) else top_label


print('[STATUS] classify_type_cascade defined (entropy/margin gate)')



# --- Code Cell ---
# [MODEL] Assembled Qwen3-VL pipeline

PARSE_FAIL_STREAK = 0
PARSE_FAIL_ABORT = 10


TEMPORAL_ANCHOR_DELTA_MAX = 3.0   # seconds; same bounded-correction idea as
                                  # Section 6.4's PE refinement, roles reversed:
                                  # here the classical (VLM-free) estimate is the
                                  # anchor and the VLM reading is the bounded
                                  # correction, so a misread timestamp cannot drag
                                  # the prediction arbitrarily far from a z-score
                                  # + OSD estimate that never saw the VLM's answer.


def run_inference_vlm(video_path: pathlib.Path, sub_path: str = None) -> dict:
    """Three staged VLM calls, then classical-signal anchoring and the entropy
    cascade, then the scene rule. No training, no tracking."""
    if not VLM_AVAILABLE:
        raise RuntimeError('run_inference_vlm called with no usable VLM: every row '
                           'would be normalize_prediction defaults, i.e. a constant.')
    cap = cv2.VideoCapture(str(video_path))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = (n / fps) if fps > 0 else 20.0

    scene = SCENE_BY_PATH.get(sub_path or ('videos/' + video_path.name))

    pred = stage1_full_scan(video_path, duration, scene)

    # A run where Stage 1 never parses is the constant-submission failure again,
    # just slower. Ten in a row means stop and look, not carry on for 8 hours.
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

    # Anchor Stage 1's time against the classical (z-score + OSD) ensemble
    # before the VLM's own bounded refinement runs, so a bad timestamp read
    # in Stage 1 is caught by a signal that never looked at pixels the VLM
    # hallucinated over -- this is the "temporal drift" failure mode.
    t_classical = predict_accident_time_ensemble(video_path)
    t_vlm = pred['accident_time']
    correction = float(np.clip(t_vlm - t_classical, -TEMPORAL_ANCHOR_DELTA_MAX,
                               TEMPORAL_ANCHOR_DELTA_MAX))
    pred['accident_time'] = float(np.clip(t_classical + correction, 0.0, duration))

    t_final = stage2_time_refine(video_path, pred['accident_time'], duration)
    pred['accident_time'] = t_final

    pt = stage3_grounding(video_path, t_final)
    if pt is not None:                      # grounding_weight = 1.0: it replaces
        pred['center_x'], pred['center_y'] = pt

    # Entropy/margin cascade: only escalates past two extra cheap calls when
    # they disagree with each other and with Stage 1 -- see classify_type_cascade.
    pred['type'] = classify_type_cascade(video_path, t_final, pred['type'])
    pred['type'] = apply_scene_type_postfix(pred['type'], scene)

    return {'path': str(video_path), 'accident_time': pred['accident_time'],
            'center_x': pred['center_x'], 'center_y': pred['center_y'],
            'type': pred['type'], 'scene_layout': scene}


print('[STATUS] run_inference_vlm defined')


# ===========================================================================
# ### 6.11 Assembled Pipelines
# 
# `run_inference_vlm` (Section 6.10) is the pipeline. `run_inference_final` below is the tracking-based design, retained so Section 7 can score all three against each other and against the constant floor; its measured result is a negative one.
# ===========================================================================


# --- Code Cell ---
# [MODEL] Baseline pipeline -- classical signals + CLIP only (kept for comparison)

def run_inference_baseline(video_path: pathlib.Path) -> dict:
    acc_time = predict_accident_time(video_path)
    cx, cy   = predict_impact_location_flow(video_path, accident_time=acc_time)
    col_type = predict_collision_type_clip(video_path, acc_time)
    return {
        'path': str(video_path), 'accident_time': acc_time,
        'center_x': cx, 'center_y': cy, 'type': col_type,
    }


# [MODEL] Final pipeline -- tracking when its own evidence supports it, constant otherwise
#
# What changed, and why:
#
#  1. The gate. Tracking used to be trusted only when its impact time agreed
#     with the frame-difference anchor. That anchor scores T=0.31 against a
#     constant's 0.52 -- it was validating tracking against a signal weaker
#     than a constant. The gate is now the tracker's own kinetic evidence.
#  2. The fallbacks are constants. Previously a distrusted video fell through to
#     OWLv2 / optical-flow (S=0.156) and the frame-difference anchor (T=0.314),
#     both below the constant floor. Falling back to a weaker model than the
#     prior subtracts score on every video it touches.
#  3. Qwen, the Perception Encoder, and OWLv2 are gone. Qwen dragged C from
#     0.30 to 0.15 (below the 0.20 chance level) by collapsing to 'rear-end' on
#     18/20 videos, and it dominated the runtime. Section 7b re-measures every
#     removed signal against its constant rather than taking this on faith.

TRACK_EVIDENCE_MIN = 0.0    # set from the Section 7b sweep before the real run

# Set by Section 6.8. None => fall back to the hand-written geometry rule, so
# the pipeline still runs end-to-end if feature extraction was skipped.
TYPE_CLASSIFIER = globals().get('TYPE_CLASSIFIER')

# Systematic offset between a tracked box-contact and the dataset's definition
# of accident_time, measured in Section 6.8. 0.0 if that section was skipped.
TIME_OFFSET = globals().get('TIME_OFFSET', 0.0)

# CLIP scores C=0.30, which beats the constant on the BALANCED calibration split
# (0.20) but loses to it under the natural synthetic prior (0.359). The real test
# prior is unknown, so this stays an explicit switch rather than a silent choice.
USE_CLIP_TYPE_FALLBACK = True


def _type_fallback(video_path: pathlib.Path, acc_time: float) -> str:
    if USE_CLIP_TYPE_FALLBACK and CLIP_AVAILABLE:
        return predict_collision_type_clip(video_path, acc_time)
    return CONST['type']


def _predict_type(coll: dict, video_path: pathlib.Path, acc_time: float) -> str:
    """Type from the supervised classifier when its features exist, else the
    hand-written geometry rule, else the zero-shot fallback.

    The trained model supersedes _type_from_geometry: the rule hard-codes angle
    bands (>=135 head-on, <=40 rear-end/sideswipe, 60-120 t-bone) and abstains
    in between, while the tree learns the same boundaries from 2211 labelled
    examples and never abstains.
    """
    feats = coll.get('features')
    if TYPE_CLASSIFIER is not None and feats is not None:
        x = np.array([[feats.get(c, np.nan) for c in FEATURE_COLS]], dtype=float)
        return str(TYPE_CLASSIFIER.predict(x)[0])
    return coll.get('type_hint') or _type_fallback(video_path, acc_time)


def run_inference_final(video_path: pathlib.Path) -> dict:
    coll = analyze_video_collision(video_path)
    trusted = coll['found'] and coll.get('evidence', 0.0) >= TRACK_EVIDENCE_MIN

    if trusted:
        acc_time = coll['t_impact'] - TIME_OFFSET
        cx, cy   = coll['center']
        col_type = _predict_type(coll, video_path, acc_time)
    else:
        acc_time = CONST['accident_time']
        cx, cy   = CONST['center_x'], CONST['center_y']
        col_type = _type_fallback(video_path, acc_time)

    return {
        'path': str(video_path), 'accident_time': acc_time,
        'center_x': cx, 'center_y': cy, 'type': col_type,
        'track_used': bool(trusted),
    }


print('[STATUS] run_inference_baseline / run_inference_final defined')


# ===========================================================================
# ---
# ## 7. Evaluation
# 
# Scoring follows the official Evaluation page: Gaussian temporal similarity (T), Gaussian spatial similarity (S), and top-1 collision-type accuracy (C). Each component is **averaged across videos first**, and the leaderboard score is the harmonic mean of those three averages:
# 
# $$\text{ACCIDENT score} = \frac{3}{\frac{1}{\bar{T}} + \frac{1}{\bar{S}} + \frac{1}{\bar{C}}}$$
# 
# **This ordering matters, and an earlier revision had it backwards.** It computed a harmonic mean per video and averaged *that*. Because C is 0/1, that variant zeroes out every misclassified video and is bounded above by accuracy -- a different and much harsher metric. It reported 0.087 for the baseline where the real metric gives **0.2321**, and 0.0000 for the final pipeline where the real metric gives **0.1739**. It also produced the false conclusion that classification dominates the score (see Section 6.5). Under the real metric the harmonic mean penalizes the **weakest** component, which for this pipeline is spatial localization.
# 
# Measured leverage on the baseline (T=0.314, S=0.156, C=0.300):
# 
# | improvement | score becomes | gain |
# |---|---|---|
# | S: 0.156 → 0.45 | 0.3433 | **+0.111** |
# | C: 0.300 → 0.45 | 0.2539 | +0.022 |
# | T: 0.314 → 0.45 | 0.2507 | +0.019 |
# 
# Spatial has roughly five times the leverage of classification -- the reverse of what this notebook was built to assume.
# 
# **The sigma values are not published.** The Evaluation page says only "Gaussian-style similarity". `SIGMA_T = 2.0` and `SIGMA_S = 0.1` are assumptions. They matter enormously: a constant spatial prediction scores 0.095 at σ=0.05 but 0.622 at σ=0.20, which inverts the priority order above. Every conclusion here is conditional on them.
# 
# **The constant-prediction floor (Section 7a).** Because σ_S is comparable to the spread of the labels themselves (center_x std 0.13, center_y std 0.18), the dataset's own prior is a strong predictor: any model whose error exceeds the label spread scores *worse than a constant*. The floor is fitted on the synthetic training set and evaluated on the calibration split. **Any component that cannot beat its own constant is subtracting value and should be replaced by that constant** -- including as a fallback, where this pipeline currently falls back from a weak model to an even weaker one.
# 
# **Calibration set caveats.** Calibration uses a diverse set of 20 videos -- 4 per collision type. This guards against the Section 7b failure (tuning on a head-on-only subset reached 0.80 there and 0.10 on the diverse set). But two limits should be stated plainly: the balanced split makes the measured C a *balanced* accuracy, which is **not** what the leaderboard measures (the real test set has its own unknown class prior); and n=20 is far too small to separate differences of ~0.15 in C, which is 3 videos.
# ===========================================================================


# --- Code Cell ---
# [EVAL] Competition scoring functions -- matches the official Evaluation page
#
# The leaderboard averages each component across videos FIRST, then takes the
# harmonic mean of those three averages:
#
#     ACCIDENT score = 3 / (1/mean(T) + 1/mean(S) + 1/mean(C))
#
# An earlier revision computed a per-video harmonic mean and averaged *that*.
# Because C is 0/1, that variant zeroes every misclassified video, so it is
# bounded above by accuracy -- a different, far harsher metric. It reported
# 0.087 for the baseline where the real metric gives 0.232, and it is what
# motivated the (false) claim that classification dominates the score.

# SIGMA_T / SIGMA_S are defined with the constant prior in Section 6.7.

def temporal_score(pred_time: float, gt_time: float) -> float:
    """Gaussian temporal similarity averaged over the three tolerances."""
    return float(np.mean([np.exp(-0.5 * ((pred_time - gt_time) / s) ** 2)
                          for s in SIGMA_T_LIST]))


def spatial_score(pred_x: float, pred_y: float, gt_x: float, gt_y: float) -> float:
    """Anisotropic Gaussian spatial similarity; sigmas are the mean bbox size."""
    return float(np.exp(-0.5 * (((pred_x - gt_x) / SIGMA_X) ** 2
                                + ((pred_y - gt_y) / SIGMA_Y) ** 2)))


def classification_score(pred_type: str, gt_type: str) -> int:
    """1 if predicted type matches ground truth, else 0."""
    return int(pred_type.strip().lower() == gt_type.strip().lower())


def accident_score(eval_df: pd.DataFrame) -> float:
    """Official leaderboard score for an already-scored prediction set."""
    T, S, C = eval_df['T'].mean(), eval_df['S'].mean(), eval_df['C'].mean()
    if min(T, S, C) <= 0:
        return 0.0
    return float(3.0 / (1.0 / T + 1.0 / S + 1.0 / C))


def score_predictions(df_pred: pd.DataFrame, labels_ref: pd.DataFrame) -> pd.DataFrame:
    """Join predictions to ground truth on video stem; return per-video T/S/C.

    Deliberately no per-video H column -- the metric is not defined per video.
    Pass the returned frame to accident_score().
    """
    df_pred = df_pred.copy()
    df_pred['video_stem'] = df_pred['path'].map(lambda p: pathlib.Path(p).stem)

    labels_ref = labels_ref.copy()
    labels_ref['video_stem'] = labels_ref['rgb_path'].map(lambda p: pathlib.Path(p).stem)

    merged = df_pred.merge(
        labels_ref[['video_stem', 'accident_time', 'center_x', 'center_y', 'type']],
        on='video_stem', how='inner', suffixes=('_pred', '_gt'),
    )
    merged['T'] = merged.apply(
        lambda r: temporal_score(r['accident_time_pred'], r['accident_time_gt']), axis=1)
    merged['S'] = merged.apply(
        lambda r: spatial_score(r['center_x_pred'], r['center_y_pred'],
                                r['center_x_gt'], r['center_y_gt']), axis=1)
    merged['C'] = merged.apply(
        lambda r: classification_score(r['type_pred'], r['type_gt']), axis=1)
    return merged


print('[STATUS] Scoring functions defined (average components, then harmonic mean)')


# --- Code Cell ---
# [EVAL] Build the diverse calibration set -- N_PER_TYPE videos per collision type
N_PER_TYPE = 4

diverse_rows = []
for coll_type in COLLISION_TYPES:
    subset = labels_clean[labels_clean['type'] == coll_type]
    n_sample = min(N_PER_TYPE, len(subset))
    if n_sample == 0:
        print(f'[WARNING] No videos found for type={coll_type}, skipping')
        continue
    diverse_rows.append(subset.sample(n_sample, random_state=SEED))
    print(f'[STATUS] {coll_type}: sampled {n_sample}/{len(subset)} videos')

diverse_labels_df = pd.concat(diverse_rows, ignore_index=True)
diverse_videos = [pathlib.Path(p) for p in diverse_labels_df['abs_video_path']]

# Every path was existence-checked during cleaning; assert anyway before spending GPU time
assert all(vp.exists() for vp in diverse_videos), 'Some calibration videos are missing on disk'
print(f'[STATUS] Diverse calibration set: {len(diverse_videos)} videos across '
      f"{diverse_labels_df['type'].nunique()} types")
display(diverse_labels_df[['rgb_path', 'type']])

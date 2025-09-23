"""
QEM preprocessing: compute ROI / Prior / Motion per video and save to .npz

- Input CSV: DATA_PATH='./saved/data/voxceleb2/info_clean.csv'
  Each line: "<abs_video_path> <label_int>"
- Output: one .npz per video under OUT_DIR, containing:
    ROI    : (T, H', W') in {0,1}
    Prior  : (T, H', W') in [0,1]
    Motion : (T, H', W') in [0,1]
  where H' = H // P, W' = W // P, P = PATCH_SIZE

Landmarks backend: face-alignment (68 pts). Switch via get_landmarks().
"""

import os, csv, math, hashlib, warnings
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# ========= config =========
DATA_PATH   = './saved/data/voxceleb2/info_clean.csv'
OUT_DIR     = '/saved/qem/voxceleb2'
PATCH_SIZE  = 16                           
GAUSS_SIGMA = 5.0                        
MOTION_WIN  = 7                          
RESIZE_LONG = 256                         
MAX_FRAMES  = None                        
SKIP_EXIST  = True                       

# ========= choose landmark backend (default: face-alignment) =========
_LM_BACKEND = 'fa'  # 'fa' | 'dlib' | 'opencv'

def _init_fa():
    # face-alignment backend (pip install face-alignment torch torchvision)
    import face_alignment
    from skimage import io  
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda' if _has_cuda() else 'cpu')
    return fa

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

_FA = None  # lazy init

def get_landmarks(img_bgr):
    """
    Return 68x2 landmarks (float) in image coordinates, or None if fail.
    Backend: face-alignment / dlib / opencv
    """
    global _FA
    if _LM_BACKEND == 'fa':
        if _FA is None:
            _FA = _init_fa()
        # face-alignment expects RGB
        import torch
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pts = _FA.get_landmarks_from_image(img_rgb)
        if pts and len(pts) > 0:
            return pts[0].astype(np.float32)  # (68,2)
        return None

    elif _LM_BACKEND == 'dlib':
        import dlib
        # you need to provide predictor path
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        if not hasattr(get_landmarks, "_dlib"):
            get_landmarks._dlib = {}
            get_landmarks._dlib["det"] = dlib.get_frontal_face_detector()
            get_landmarks._dlib["pred"] = dlib.shape_predictor(predictor_path)
        dets = get_landmarks._dlib["det"](img_bgr, 1)
        if len(dets) == 0: return None
        shape = get_landmarks._dlib["pred"](img_bgr, dets[0])
        pts = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)
        return pts

    elif _LM_BACKEND == 'opencv':
        # requires opencv-contrib-python and lbfmodel.yaml
        if not hasattr(get_landmarks, "_cv"):
            get_landmarks._cv = {}
            get_landmarks._cv["det"] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            facemark = cv2.face.createFacemarkLBF()
            facemark.loadModel('lbfmodel.yaml')
            get_landmarks._cv["facemark"] = facemark
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rects = get_landmarks._cv["det"].detectMultiScale(gray, 1.3, 5)
        if len(rects) == 0: return None
        ok, landmarks = get_landmarks._cv["facemark"].fit(img_bgr, rects)
        if ok and len(landmarks) > 0:
            return landmarks[0][0].astype(np.float32)
        return None
    else:
        raise ValueError("Unknown landmark backend")

# iBUG-68 index（0-based）
IDX = {
    "jaw":       list(range(0,17)),
    "brow_l":    list(range(17,22)),
    "brow_r":    list(range(22,27)),
    "nose":      list(range(27,36)),
    "eye_l":     list(range(36,42)),
    "eye_r":     list(range(42,48)),
    "mouth_all": list(range(48,68)),
}

# key region
ROI_GROUPS = ["brow_l", "brow_r", "eye_l", "eye_r", "mouth_all"]

def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        if MAX_FRAMES and len(frames) >= MAX_FRAMES:
            break
    cap.release()
    if len(frames) == 0:
        return None

    if RESIZE_LONG is not None:
        out = []
        for f in frames:
            h, w = f.shape[:2]
            scale = RESIZE_LONG / max(h, w)
            if scale != 1.0:
                f = cv2.resize(f, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            out.append(f)
        frames = out
    return frames  # list of HxWx3 (BGR)

def polygon_from_points(pts):
    return np.round(pts).astype(np.int32)

def draw_roi_mask(h, w, landmarks):
    mask = np.zeros((h, w), dtype=np.uint8)
    for key in ["brow_l", "brow_r", "eye_l", "eye_r", "mouth_all"]:
        poly = polygon_from_points(landmarks[IDX[key]])
        cv2.fillPoly(mask, [poly], 1)
    return mask  # 0/1

def draw_prior_map(h, w, landmarks, sigma=GAUSS_SIGMA):
    prior = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    two_sigma2 = 2*(sigma**2)
    for key in ROI_GROUPS:
        for (x0, y0) in landmarks[IDX[key]]:
            gauss = np.exp(-((xx - x0)**2 + (yy - y0)**2)/two_sigma2)
            prior += gauss
    if prior.max() > 0:
        prior /= (prior.max() + 1e-8)
    return prior  # [0,1]

def motion_map(prev_gray, curr_gray, win=MOTION_WIN):
    diff = cv2.absdiff(curr_gray, prev_gray)
    k = win | 1
    sm = cv2.boxFilter(diff, ddepth=-1, ksize=(k, k), normalize=True)
    sm = sm.astype(np.float32)
    if sm.max() > 0:
        sm /= (sm.max() + 1e-8)
    return sm  # [0,1]

def to_patch_grid(img, P):
    """downsampling to patch size (H', W')"""
    h, w = img.shape[:2]
    Hs, Ws = h // P, w // P
    if Hs < 1 or Ws < 1:
        raise ValueError(f"PATCH_SIZE={P} too large for frame {h}x{w}")
    # 对 ROI（二值）用 INTER_AREA 下采样再阈值
    if img.dtype == np.uint8:  # ROI mask 0/1
        small = cv2.resize(img, (Ws, Hs), interpolation=cv2.INTER_AREA)
        return (small >= 0.5).astype(np.uint8)
    else:
        small = cv2.resize(img, (Ws, Hs), interpolation=cv2.INTER_AREA)
        small = np.clip(small, 0.0, 1.0)
        return small.astype(np.float32)

def process_video_one(path_mp4, out_dir):
    frames = read_video(path_mp4)
    if frames is None:
        return False, "read_fail"

    T = len(frames)
    h, w = frames[0].shape[:2]
    Hs, Ws = h // PATCH_SIZE, w // PATCH_SIZE
    ROI = np.zeros((T, Hs, Ws), dtype=np.uint8)
    Prior = np.zeros((T, Hs, Ws), dtype=np.float32)
    Motion = np.zeros((T, Hs, Ws), dtype=np.float32)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    for t, frame in enumerate(frames):
        lm = get_landmarks(frame)
        if lm is not None:
            roi_full = draw_roi_mask(h, w, lm)                   # 0/1
            prior_full = draw_prior_map(h, w, lm, GAUSS_SIGMA)   # [0,1]
        else:
            roi_full   = np.zeros((h, w), dtype=np.uint8)
            prior_full = np.zeros((h, w), dtype=np.float32)

        # Motion：start from t2
        if t == 0:
            mot_full = np.zeros((h, w), dtype=np.float32)
        else:
            mot_full = motion_map(grays[t-1], grays[t], win=MOTION_WIN)

        ROI[t]   = to_patch_grid(roi_full,   PATCH_SIZE)
        Prior[t] = to_patch_grid(prior_full, PATCH_SIZE)
        Motion[t]= to_patch_grid(mot_full,   PATCH_SIZE)

    rel_hash = hashlib.md5(path_mp4.encode('utf-8')).hexdigest()[:10]
    stem = Path(path_mp4).stem + f"_{rel_hash}"
    out_path = Path(out_dir) / (stem + ".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ROI=ROI, Prior=Prior, Motion=Motion,
                        patch_size=PATCH_SIZE, sigma=GAUSS_SIGMA, motion_win=MOTION_WIN,
                        src=path_mp4, shape=(T, h, w))
    return True, str(out_path)

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=' ')
        rows = list(reader)

    pbar = tqdm(rows, desc="Preprocessing videos")
    ok, fail = 0, 0
    for row in pbar:
        if len(row) == 0: continue
        mp4 = row[0]
        rel_hash = hashlib.md5(mp4.encode('utf-8')).hexdigest()[:10]
        stem = Path(mp4).stem + f"_{rel_hash}"
        out_path = Path(OUT_DIR) / (stem + ".npz")
        if SKIP_EXIST and out_path.exists():
            ok += 1
            pbar.set_postfix(ok=ok, fail=fail, note="skip_exist")
            continue

        try:
            succ, info = process_video_one(mp4, OUT_DIR)
            if succ:
                ok += 1
            else:
                fail += 1
            pbar.set_postfix(ok=ok, fail=fail, last=Path(info).name if succ else info)
        except Exception as e:
            fail += 1
            pbar.set_postfix(ok=ok, fail=fail, err=str(e)[:40])

    print(f"Done. ok={ok}, fail={fail}, out_dir={OUT_DIR}")

if __name__ == "__main__":
    main()

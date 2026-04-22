from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import math
import time

MODEL_PATH     = "space_v3.pt"
VIDEO_PATH     = "4.mp4"
OUTPUT_DIR     = "test_results"
CONF           = 0.25
IOU            = 0.6
TRAJECTORY_LEN = 30
COLLISION_DIST = 120

# ══════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════
COLORS = {
    "Satellite":  (0,   212, 255),   
    "Space-Rock": (185, 110, 255),   
}
ALERT_COLOR   = (0,  60, 255)        
ALERT_GLOW    = (0,  30, 180)
BG_PANEL      = (10, 14, 26)
BORDER_COLOR  = (28, 48, 80)
ACCENT_CYAN   = (0,  212, 255)
ACCENT_VIOLET = (185,110, 255)
TEXT_PRIMARY  = (220,235, 255)
TEXT_DIM      = (85, 110, 155)
GRID_COLOR    = (16,  24,  42)
WARNING_AMBER = (255, 180,  30)


# ══════════════════════════════════════════════════
# DRAWING PRIMITIVES
# ══════════════════════════════════════════════════

def glass_rect(frame, x1, y1, x2, y2,
               fill=BG_PANEL, border=BORDER_COLOR,
               fill_alpha=0.85, border_w=1):
    h, w = frame.shape[:2]
    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)
    if x2 <= x1 or y2 <= y1:
        return
    ov = frame.copy()
    cv2.rectangle(ov, (x1,y1), (x2,y2), fill, -1)
    cv2.addWeighted(ov, fill_alpha, frame, 1-fill_alpha, 0, frame)
    cv2.rectangle(frame, (x1,y1), (x2,y2), border, border_w)

def corner_brackets(frame, x1, y1, x2, y2, color, L=14, T=2):
    for cx, cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        dx = L if cx == x1 else -L
        dy = L if cy == y1 else -L
        cv2.line(frame, (cx,cy), (cx+dx,cy), color, T, cv2.LINE_AA)
        cv2.line(frame, (cx,cy), (cx,cy+dy), color, T, cv2.LINE_AA)

def glow_circle(frame, cx, cy, r, color, alpha=0.35):
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), r, color, 1, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

def txt(frame, text, pos, scale, color, thick=1, shadow=True):
    if shadow:
        cv2.putText(frame, text, (pos[0]+1,pos[1]+1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+1, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def progress_bar(frame, x, y, w, h, val, max_val, color, bg=(18,26,46)):
    cv2.rectangle(frame, (x,y), (x+w,y+h), bg, -1)
    fill = int(w * min(val, max_val) / max(max_val,1))
    if fill > 0:
        cv2.rectangle(frame, (x,y), (x+fill,y+h), color, -1)
        ov = frame.copy()
        cv2.rectangle(ov, (x+max(0,fill-4),y-1),
                      (x+fill+2, y+h+1), color, -1)
        cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)
    cv2.rectangle(frame, (x,y), (x+w,y+h), BORDER_COLOR, 1)

def bg_grid(frame):
    h, w = frame.shape[:2]
    ov = frame.copy()
    for x in range(0, w, 60):
        cv2.line(ov, (x,0), (x,h), GRID_COLOR, 1)
    for y in range(0, h, 60):
        cv2.line(ov, (0,y), (w,y), GRID_COLOR, 1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

def scanlines(frame):
    h, w = frame.shape[:2]
    sweep = int(time.time() * 120) % h
    ov = frame.copy()
    cv2.line(ov, (0, sweep), (w, sweep), ACCENT_CYAN, 1)
    cv2.addWeighted(ov, 0.07, frame, 0.93, 0, frame)
    ov2 = frame.copy()
    for y in range(0, h, 4):
        cv2.line(ov2, (0,y), (w,y), (0,0,0), 1)
    cv2.addWeighted(ov2, 0.04, frame, 0.96, 0, frame)

def dashed_line(frame, p1, p2, color, dash=8, gap=5, thick=1):
    x1,y1 = p1; x2,y2 = p2
    dist = math.hypot(x2-x1, y2-y1)
    if dist == 0: return
    dx,dy = (x2-x1)/dist, (y2-y1)/dist
    d = 0; draw = True
    while d < dist:
        end = min(d + (dash if draw else gap), dist)
        if draw:
            cv2.line(frame,
                     (int(x1+dx*d),int(y1+dy*d)),
                     (int(x1+dx*end),int(y1+dy*end)),
                     color, thick, cv2.LINE_AA)
        d = end; draw = not draw


# ══════════════════════════════════════════════════
# TRACKER
# ══════════════════════════════════════════════════
class Tracker:
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=TRAJECTORY_LEN))
        self.labels  = {}

    def update(self, tid, center, label):
        self.history[tid].append(center)
        self.labels[tid] = label

    def draw(self, frame):
        for tid, pts in self.history.items():
            pts_list = list(pts)
            color = COLORS.get(self.labels.get(tid,""), (255,255,255))
            for i in range(1, len(pts_list)):
                t = i / len(pts_list)
                c = tuple(int(v*t) for v in color)
                thick = 2 if t > 0.6 else 1
                cv2.line(frame, pts_list[i-1], pts_list[i],
                         c, thick, cv2.LINE_AA)
            if len(pts_list) > 1:
                glow_circle(frame, pts_list[-1][0], pts_list[-1][1],
                            7, color, 0.3)

    def clear(self):
        self.history.clear()
        self.labels.clear()


# ══════════════════════════════════════════════════
# STABLE ID MAPPER
# ByteTrack 
# ══════════════════════════════════════════════════
class StableIDMapper:
    CLASS_CFG = {
        "Satellite":  {"max_dist": 80,  "max_lost": 45},
        "Space-Rock": {"max_dist": 200, "max_lost": 120},  # زود الاتنين
        }
    DEFAULT_CFG = {"max_dist": 120, "max_lost": 60}

    def __init__(self):
        self.raw_to_stable = {}   # raw_id  → stable_id
        self.stable_last   = {}   # stable_id → {"cx","cy","label","frame"}
        self.next_stable   = 1

    def _cfg(self, label):
        return self.CLASS_CFG.get(label, self.DEFAULT_CFG)

    def update(self, detections, frame_n):
        # 1. expire lost stable IDs (use per-class max_lost)
        for sid in list(self.stable_last):
            info = self.stable_last[sid]
            max_lost = self._cfg(info["label"])["max_lost"]
            if frame_n - info["frame"] > max_lost:
                del self.stable_last[sid]

        # 2. clean raw→stable entries whose stable expired
        self.raw_to_stable = {
            r: s for r, s in self.raw_to_stable.items()
            if s in self.stable_last
        }

        for det in detections:
            raw   = det["id"]
            label = det["label"]
            cfg   = self._cfg(label)

            # already known raw id → just refresh
            if raw in self.raw_to_stable:
                sid = self.raw_to_stable[raw]
                self.stable_last[sid] = {
                    "cx": det["cx"], "cy": det["cy"],
                    "label": label, "frame": frame_n
                }
                det["id"] = sid
                continue

            # new raw id → match by proximity + same label
            best_sid, best_dist = None, cfg["max_dist"]
            for sid, info in self.stable_last.items():
                if info["label"] != label:
                    continue
                d = math.hypot(det["cx"]-info["cx"], det["cy"]-info["cy"])
                if d < best_dist:
                    best_dist = d
                    best_sid  = sid

            if best_sid is not None:
                self.raw_to_stable[raw] = best_sid
                self.stable_last[best_sid] = {
                    "cx": det["cx"], "cy": det["cy"],
                    "label": label, "frame": frame_n
                }
                det["id"] = best_sid
            else:
                sid = self.next_stable
                self.next_stable += 1
                self.raw_to_stable[raw] = sid
                self.stable_last[sid] = {
                    "cx": det["cx"], "cy": det["cy"],
                    "label": label, "frame": frame_n
                }
                det["id"] = sid


# ══════════════════════════════════════════════════
# COLLISION CHECK
# ══════════════════════════════════════════════════
def check_collision(detections):
    alerts = []
    sats  = [d for d in detections if d["label"] == "Satellite"]
    rocks = [d for d in detections if d["label"] == "Space-Rock"]
    for s in sats:
        for r in rocks:
            dist = np.hypot(s["cx"]-r["cx"], s["cy"]-r["cy"])
            if dist < COLLISION_DIST:
                alerts.append((s["id"], r["id"], int(dist)))
    return alerts


# ══════════════════════════════════════════════════
# DETECTION BOX
# ══════════════════════════════════════════════════
def draw_box(frame, det, in_alert):
    x1,y1,x2,y2 = det["box"]
    label = det["label"]
    tid   = det["id"]
    conf  = det["conf"]
    cx,cy = det["cx"], det["cy"]
    color = ALERT_COLOR if in_alert else COLORS.get(label,(255,255,255))

    pad = 5
    bx1,by1,bx2,by2 = x1-pad,y1-pad,x2+pad,y2+pad

    # Tinted fill
    ov = frame.copy()
    cv2.rectangle(ov,(bx1,by1),(bx2,by2),
                  tuple(int(c*0.10) for c in color),-1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    # Bracket corners
    corner_brackets(frame, bx1,by1,bx2,by2, color, L=13, T=2)

    # Center crosshair
    cv2.line(frame,(cx-7,cy),(cx+7,cy),color,1,cv2.LINE_AA)
    cv2.line(frame,(cx,cy-7),(cx,cy+7),color,1,cv2.LINE_AA)

    # Outer ring
    glow_circle(frame, cx, cy, max(16,(bx2-bx1)//2+10), color, 0.28)

    # Alert pulsing ring
    if in_alert:
        pulse = int(time.time()*4)%2
        r2 = max(22,(bx2-bx1)//2+18)
        glow_circle(frame, cx, cy, r2, ALERT_COLOR, 0.5 if pulse else 0.2)

    # Label tag  (no Unicode — cv2 doesn't support it)
    short = "SAT" if label == "Satellite" else "ROCK"
    tag   = f"{short}-{tid:02d}"
    conf_str = f"{conf*100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (lw,lh),_ = cv2.getTextSize(tag,      font, 0.40, 1)
    (cw,_),_  = cv2.getTextSize(conf_str, font, 0.30, 1)
    tw = lw + cw + 16
    tx = max(0, bx1)
    ty = max(16, by1-5)
    glass_rect(frame, tx-3, ty-lh-6, tx+tw+2, ty+3,
               fill=BG_PANEL, border=color, fill_alpha=0.88)
    cv2.circle(frame, (tx+4, ty-lh//2-1), 3, color, -1)
    txt(frame, tag,      (tx+12, ty-1), 0.40, TEXT_PRIMARY)
    txt(frame, conf_str, (tx+12+lw+5, ty-1), 0.30, color)


# ══════════════════════════════════════════════════
# COLLISION THREAT LINE
# ══════════════════════════════════════════════════
def draw_threat_line(frame, detections, alerts):
    id_map = {d["id"]: d for d in detections}
    for sid, rid, dist in alerts:
        if sid not in id_map or rid not in id_map:
            continue
        s = id_map[sid]; r = id_map[rid]
        # Dashed threat line
        pulse = int(time.time()*6)%2
        col = ALERT_COLOR if pulse else WARNING_AMBER
        dashed_line(frame, (s["cx"],s["cy"]), (r["cx"],r["cy"]),
                    col, dash=10, gap=6, thick=2)
        # Midpoint danger badge
        mx = (s["cx"]+r["cx"])//2
        my = (s["cy"]+r["cy"])//2
        badge = f"! {dist}px"
        (bw,bh),_ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        glass_rect(frame, mx-bw//2-6, my-bh-6, mx+bw//2+6, my+4,
                   fill=(30,0,0), border=ALERT_COLOR, fill_alpha=0.92)
        txt(frame, badge, (mx-bw//2, my), 0.38, WARNING_AMBER)


# ══════════════════════════════════════════════════
# WARNING BANNER (bottom)
# ══════════════════════════════════════════════════
def draw_warning_banner(frame, alerts):
    if not alerts:
        return
    h, w = frame.shape[:2]
    n = len(alerts)
    band_h = 30 * n + 10

    # Flashing red background
    pulse = int(time.time()*4)%2
    alpha = 0.82 if pulse else 0.60
    ov = frame.copy()
    cv2.rectangle(ov,(0,h-band_h),(w,h),(12,0,0),-1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    cv2.line(frame,(0,h-band_h),(w,h-band_h),ALERT_COLOR,2)

    for i,(sid,rid,dist) in enumerate(alerts):
        y = h - band_h + 22 + i*30
        icon_col = WARNING_AMBER if int(time.time()*4)%2 else ALERT_COLOR
        txt(frame, "!!", (12, y), 0.46, icon_col, 2)
        msg = f"COLLISION  SAT-{sid:02d} <-> ROCK-{rid:02d}  DIST:{dist}px"
        txt(frame, msg, (40, y), 0.40, TEXT_PRIMARY, 1)
        danger = max(0, 1 - dist/COLLISION_DIST)
        bar_w = min(180, w//5)
        progress_bar(frame, w-bar_w-44, y-11, bar_w, 6,
                     int(danger*100), 100,
                     ALERT_COLOR, bg=(30,5,5))
        txt(frame, f"{danger*100:.0f}%", (w-40,y), 0.30, ALERT_COLOR)


# ══════════════════════════════════════════════════
# TOP-LEFT HEADER
# ══════════════════════════════════════════════════
def draw_header(frame, frame_n, fps, n_sat, n_rock, n_alert):
    glass_rect(frame, 6,6, 270,56, fill=BG_PANEL, border=BORDER_COLOR)
    cv2.line(frame, (6,7), (270,7), ACCENT_CYAN, 2)
    txt(frame, "SPACE SURVEILLANCE", (14,24), 0.42, ACCENT_CYAN)
    t_str = f"T+{frame_n/max(fps,1):.1f}s   F{frame_n}"
    txt(frame, t_str, (14,42), 0.30, TEXT_DIM)

    # Mini stat chips
    chips = [
        (f"SAT {n_sat}",   ACCENT_CYAN,   276),
        (f"ROCK {n_rock}", ACCENT_VIOLET, 324),
    ]
    for label, col, x in chips:
        (cw,ch),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
        glass_rect(frame, x,10, x+cw+10,30,
                   fill=tuple(int(c*0.12) for c in col),
                   border=col, fill_alpha=0.9)
        txt(frame, label, (x+5,25), 0.30, col)


# ══════════════════════════════════════════════════
# RIGHT PANEL
# ══════════════════════════════════════════════════
def draw_side_panel(frame, detections, alert_frames, frame_n, fps):
    h, w = frame.shape[:2]
    sats  = [d for d in detections if d["label"] == "Satellite"]
    rocks = [d for d in detections if d["label"] == "Space-Rock"]
    rows  = len(sats) + len(rocks)

    pw = 210
    ph = 96 + rows * 34 + 10
    px = w - pw - 8
    py = 8

    glass_rect(frame, px,py, px+pw,py+ph, fill=BG_PANEL, border=BORDER_COLOR)
    cv2.line(frame, (px,py+1), (px+pw,py+1), ACCENT_CYAN, 2)

    txt(frame,"OBJECT REGISTRY",(px+10,py+18),0.38,ACCENT_CYAN)

    # Blink dot
    pulse = int(time.time()*2)%2
    cv2.circle(frame,(px+pw-12,py+13),4,
               (0,210,80) if pulse else (0,80,30),-1)

    cv2.line(frame,(px+6,py+24),(px+pw-6,py+24),BORDER_COLOR,1)

    # Stats
    t_sec = frame_n/max(fps,1)
    stats = [("TIME",f"T+{t_sec:.1f}s"),
             ("SAT", str(len(sats))),
             ("ROCK",str(len(rocks))),
             ("ALRT",str(alert_frames))]
    col_w = pw//4
    for i,(lbl,val) in enumerate(stats):
        sx = px + i*col_w + 5
        txt(frame, lbl, (sx,py+38), 0.24, TEXT_DIM)
        vc = ALERT_COLOR if lbl=="ALRT" and alert_frames>0 else TEXT_PRIMARY
        txt(frame, val,  (sx,py+52), 0.36, vc)

    cv2.line(frame,(px+6,py+60),(px+pw-6,py+60),BORDER_COLOR,1)
    txt(frame,"ID  TYPE  CONF",
        (px+8,py+74),0.24,TEXT_DIM)

    row_y = py+86
    for det in sats + rocks:
        label = det["label"]
        color = COLORS.get(label,(255,255,255))
        cv2.rectangle(frame,(px+6,row_y-2),(px+11,row_y+8),color,-1)
        txt(frame,f"#{det['id']:02d}", (px+15,row_y+8),0.32,TEXT_PRIMARY)

        short = "SAT" if label=="Satellite" else "ROCK"
        txt(frame,short,(px+48,row_y+8),0.30,color)

        txt(frame,f"{det['conf']*100:.0f}%",(px+94,row_y+8),0.30,TEXT_DIM)

        # Conf bar
        progress_bar(frame,px+6,row_y+12,pw-16,3,
                     int(det["conf"]*100),100,color)
        row_y += 34


# ══════════════════════════════════════════════════
# MAIN HUD
# ══════════════════════════════════════════════════
def draw_hud(frame, detections, alerts, alert_frames, frame_n, fps):
    bg_grid(frame)

    # Threat lines between colliding pairs
    draw_threat_line(frame, detections, alerts)

    alert_ids = {a[0] for a in alerts} | {a[1] for a in alerts}
    for det in detections:
        draw_box(frame, det, det["id"] in alert_ids)

    scanlines(frame)

    n_sat  = sum(1 for d in detections if d["label"]=="Satellite")
    n_rock = sum(1 for d in detections if d["label"]=="Space-Rock")
    draw_header(frame, frame_n, fps, n_sat, n_rock, len(alerts))
    draw_side_panel(frame, detections, alert_frames, frame_n, fps)
    draw_warning_banner(frame, alerts)


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
def run(video_path=VIDEO_PATH, conf=CONF, iou=IOU, save_frames=True):
    out_dir    = Path(OUTPUT_DIR) / "video"
    frames_dir = out_dir / "alert_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_frames:
        frames_dir.mkdir(exist_ok=True)

    model    = YOLO(MODEL_PATH)
    tracker  = Tracker()
    id_mapper = StableIDMapper()
    cap      = cv2.VideoCapture(video_path)

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(out_dir/"output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    print(f"Video : {video_path}  ({width}x{height} | {fps}fps | {total} frames)")
    print(f"Model : {MODEL_PATH}  |  conf={conf}  iou={iou}")
    print("Press Q to quit\n")

    frame_n = sat_total = rock_total = alert_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1

        results = model.track(
            frame, persist=True, conf=conf, iou=iou,
            tracker="bytetrack.yaml", verbose=False,
        )

        detections = []
        r = results[0]
        if r.boxes is not None and r.boxes.id is not None:
            for box, cf, cls, tid in zip(
                r.boxes.xyxy.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.id.cpu().numpy().astype(int),
            ):
                x1,y1,x2,y2 = box
                cx,cy = (x1+x2)//2, (y1+y2)//2
                label = r.names[cls]
                detections.append({
                    "id":int(tid),"label":label,"conf":float(cf),
                    "box":(x1,y1,x2,y2),"cx":cx,"cy":cy,
                })

        # stabilise IDs before any drawing/tracking
        id_mapper.update(detections, frame_n)

        for det in detections:
            tracker.update(det["id"], (det["cx"], det["cy"]), det["label"])

        tracker.draw(frame)
        alerts = check_collision(detections)

        if alerts:
            alert_frames += 1
            if save_frames:
                cv2.imwrite(str(frames_dir/f"alert_{frame_n:06d}.jpg"), frame)

        for det in detections:
            if det["label"] == "Satellite": sat_total  += 1
            else:                           rock_total += 1

        draw_hud(frame, detections, alerts, alert_frames, frame_n, fps)

        writer.write(frame)
        cv2.imshow("Space Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if frame_n % 100 == 0:
            live = len(detections)
            print(f"[{frame_n}/{total}] {frame_n/total*100:.1f}%  "
                  f"SAT:{sat_total}  ROCK:{rock_total}  ALERTS:{alert_frames}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\nDone — {frame_n} frames")
    print(f"Satellites  : {sat_total}")
    print(f"Space-Rocks : {rock_total}")
    print(f"Alert frames: {alert_frames}")
    print(f"Saved       : {out_dir}/output.mp4")


if __name__ == "__main__":
    run()
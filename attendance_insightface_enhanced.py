"""
attendance_insightface_enhanced.py
- Uses insightface FaceAnalysis (buffalo_l)
- Preprocesses small/shaded faces (CLAHE + upsample)
- Temporal aggregation (frames per track)
- FPS counter
- GUI overlay (name + distance + quality)
- Face quality score gating
"""

import os, time, sqlite3
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

DB_PATH = "attendance.db"

# --------------------------
# DB helpers
# --------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 roll TEXT UNIQUE, name TEXT, embedding BLOB)""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 student_id INTEGER,
                 ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                 status TEXT,
                 FOREIGN KEY(student_id) REFERENCES students(id))""")
    conn.commit(); conn.close()


def add_student(roll, name, emb: np.ndarray):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO students (roll,name,embedding) VALUES (?,?,?)",
              (roll, name, emb.astype(np.float32).tobytes()))
    conn.commit(); conn.close()


def load_registry():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, roll, name, embedding FROM students")
    rows = c.fetchall(); conn.close()
    reg = []
    for r in rows:
        sid, roll, name, emb_blob = r
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        reg.append({'id': sid, 'roll': roll, 'name': name, 'emb': emb})
    return reg


def log_attendance(student_id: int, status="present"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO attendance (student_id, status) VALUES (?,?)",
              (student_id, status))
    conn.commit(); conn.close()


# --------------------------
# Face Quality Score
# --------------------------
def face_quality(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # brightness
    brightness = gray.mean()

    # sharpness using Laplacian
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # size
    h, w = face_roi.shape[:2]

    # combined quality score 0–1
    qscore = (
        (brightness / 255) * 0.4 +
        (min(w, h) / 112) * 0.3 +
        (min(sharpness, 200) / 200) * 0.3
    )

    return qscore, brightness, sharpness


QUALITY_THRESHOLD = 0.45  # minimum quality to allow recognition

# --------------------------
# Preprocessing + Tracker
# --------------------------
UPSCALE_TO = (112, 112)
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
EMB_FRAMES = 6
TRACK_MAX_INACTIVE = 2.0
CENTROID_MATCH_DIST = 80
COSINE_THRESHOLD = 0.45

_tracks = {}
_next_track_id = 0


def apply_clahe(face_bgr):
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_face(face_bgr):
    enhanced = apply_clahe(face_bgr)
    up = cv2.resize(enhanced, UPSCALE_TO, interpolation=cv2.INTER_CUBIC)
    return up


def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def match_track(ct):
    best = None
    bd = 1e9
    for tid, t in _tracks.items():
        tx, ty = t['centroid']
        d = np.hypot(tx - ct[0], ty - ct[1])
        if d < bd:
            bd = d; best = tid
    return best if bd <= CENTROID_MATCH_DIST else None


def prune_tracks():
    now = time.time()
    rem = [tid for tid, t in _tracks.items()
           if now - t['last_seen'] > TRACK_MAX_INACTIVE]
    for tid in rem:
        del _tracks[tid]


# --------------------------
# Registration
# --------------------------
def register_from_cam(app, roll, name, captures=8, cam_index=1):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    captured = []

    print("Press SPACE to capture face (need {} captures). Press q to quit.".format(captures))

    while len(captured) < captures:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Register - press SPACE", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord(' '):
            faces = app.get(frame)
            if len(faces) > 0 and hasattr(faces[0], 'embedding'):
                emb = faces[0].embedding
                if emb is not None:
                    captured.append(emb)
                    print("Captured", len(captured))
                else:
                    print("No embedding, try again.")
            else:
                print("No face detected.")
        elif k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not captured:
        print("No captures — registration failed.")
        return

    avg = np.mean(np.stack(captured), axis=0).astype(np.float32)
    add_student(roll, name, avg)
    print("Registered:", name)


# --------------------------
# MAIN CAMERA LOOP
# --------------------------
def run_camera(app, cam_index=1):
    global _next_track_id, _tracks

    init_db()
    registry = load_registry()
    last_mark = {}

    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)

    print("Starting camera. Registered students:", len(registry))

    fps_prev = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS
        now = time.time()
        fps = 1.0 / (now - fps_prev)
        fps_prev = now

        dets = app.get(frame)
        prune_tracks()

        for f in dets:
            bbox = f.bbox.astype(int)
            x1,y1,x2,y2 = bbox
            h, w = frame.shape[:2]

            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(w - 1, x2); y2c = min(h - 1, y2)

            if x2c - x1c <= 0 or y2c - y1c <= 0:
                continue

            face_roi = frame[y1c:y2c, x1c:x2c].copy()
            face_proc = preprocess_face(face_roi)

            # --- quality check ---
            qscore, bright, sharp = face_quality(face_roi)
            if qscore < QUALITY_THRESHOLD:
                cv2.putText(frame, f"LowQ {qscore:.2f}", (x1c, y1c - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                continue

            # get embedding
            res = app.get(face_proc)
            emb = None
            if len(res) > 0 and hasattr(res[0], 'embedding') and res[0].embedding is not None:
                emb = res[0].embedding
            else:
                emb = getattr(f, 'embedding', None)

            if emb is None:
                continue

            c = centroid(bbox)
            matched = match_track(c)

            if matched is None:
                tid = _next_track_id; _next_track_id += 1
                _tracks[tid] = {'centroid': c, 'embs': [emb], 'last_seen': time.time()}
            else:
                tid = matched
                t = _tracks[tid]
                t['centroid'] = c
                t['embs'].append(emb)
                t['last_seen'] = time.time()

            t = _tracks[tid]

            # enough frames?
            if len(t['embs']) >= EMB_FRAMES:
                avg = np.mean(np.stack(t['embs'][-EMB_FRAMES:]), axis=0)
                best = None
                best_score = 1.0

                for r in registry:
                    d = cosine(avg, r['emb'])
                    if d < best_score:
                        best_score = d; best = r

                if best and best_score < COSINE_THRESHOLD:
                    sid = best['id']

                    # draw recognized overlay
                    label = f"{best['name']} ({best_score:.3f})"
                    cv2.putText(frame, label, (x1c, y1c - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                    now = time.time()
                    if sid not in last_mark or now - last_mark[sid] > 60:
                        log_attendance(sid, "present")
                        last_mark[sid] = now

                    t['embs'].clear()

                else:
                    cv2.putText(frame, f"NoMatch {best_score:.3f}", (x1c, y1c - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    t['embs'].clear()

            cv2.rectangle(frame, (x1c,y1c), (x2c,y2c), (0,255,0), 2)

        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Attendance (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------
# CLI
# --------------------------
def main():
    init_db()
    print("Insightface version:", __import__("insightface").__version__)

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1)

    print("Model ready.")
    print("Choose action: [r]egister , [s]tart camera")

    c = input("Action (r/s): ").strip().lower()

    if c == 'r':
        roll = input("Roll: ").strip()
        name = input("Name: ").strip()
        register_from_cam(app, roll, name, cam_index=1)
    else:
        run_camera(app, cam_index=1)


if __name__ == "__main__":
    main()

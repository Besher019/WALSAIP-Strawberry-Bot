#!/usr/bin/env python3
from flask import (
    Flask, request, jsonify, render_template_string, send_from_directory, abort
)
from datetime import datetime
import os
import uuid
import json

app = Flask(__name__)

# Load YOLO model globally (assuming it's available; adjust path as needed)
try:
    from ultralytics import YOLO
    MODEL = YOLO(r"weights\best.pt")# Load a custom YOLO model
    print("[SERVER] YOLO model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO: {e}")
    MODEL = None

# ============================================================
# CONFIG  (absolute paths so saving/serving always matches)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SAVE_DIR = os.path.join(BASE_DIR, "received_photos")
MISSIONS_DIR = os.path.join(BASE_DIR, "missions")

os.makedirs(BASE_SAVE_DIR, exist_ok=True)
os.makedirs(MISSIONS_DIR, exist_ok=True)

# ============================================================
# GLOBAL STATE
# ============================================================

MISSION_STATE = "idle"              # "idle" or "start"
LAST_STATUS = {}                    # last status JSON from robot
CURRENT_MISSION_ID = None           # e.g. "mission_20251210_123456_ABCD12"
MISSIONS = {}                       # mission_id -> mission dict
CONTROL_COMMAND = "none"            # "none" | "abort" | "go_home"
LAST_TERMINAL_STATUS = None         # "mission_complete" | "home_unreachable" | "mission_aborted_by_operator"


# ============================================================
# HELPERS: MISSIONS
# ============================================================

def _new_mission_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6].upper()
    return f"mission_{ts}_{short}"


def _start_new_mission() -> str:
    """Create and register a new mission, set as current."""
    global CURRENT_MISSION_ID, MISSIONS, LAST_TERMINAL_STATUS

    mission_id = _new_mission_id()
    now = datetime.now().isoformat(timespec="seconds")

    MISSIONS[mission_id] = {
        "id": mission_id,
        "started_at": now,
        "ended_at": None,
        "status": "pending",  # pending|running|complete|home_unreachable|aborted
        "waypoints_reached": set(),
        "waypoints_unreachable": set(),
        "images_count": 0,
        "last_status": None,
    }
    CURRENT_MISSION_ID = mission_id
    LAST_TERMINAL_STATUS = None

    print(f"[SERVER] New mission created: {mission_id}")
    return mission_id


def _update_mission_on_status(status_payload: dict):
    """Update mission data based on status from the robot."""
    global CURRENT_MISSION_ID, MISSIONS, LAST_TERMINAL_STATUS

    if not CURRENT_MISSION_ID:
        if status_payload.get("status") == "mission_started":
            _start_new_mission()
        else:
            return

    mission = MISSIONS.get(CURRENT_MISSION_ID)
    if mission is None:
        return

    mission["last_status"] = status_payload
    status = status_payload.get("status", "")
    now = datetime.now().isoformat(timespec="seconds")

    if status == "mission_started":
        mission["status"] = "running"

    if status == "waypoint_reached":
        idx = status_payload.get("index")
        if idx is not None:
            mission["waypoints_reached"].add(int(idx))
    elif status == "waypoint_unreachable":
        idx = status_payload.get("index")
        if idx is not None:
            mission["waypoints_unreachable"].add(int(idx))

    if status in ("mission_complete", "mission_complete_after_abort"):
        mission["status"] = "complete"
        mission["ended_at"] = now
        LAST_TERMINAL_STATUS = "mission_complete"

    elif status.startswith("home_unreachable"):
        mission["status"] = "home_unreachable"
        mission["ended_at"] = now
        LAST_TERMINAL_STATUS = "home_unreachable"

    elif status == "mission_aborted_by_operator":
        mission["status"] = "aborted"
        mission["ended_at"] = now
        LAST_TERMINAL_STATUS = "mission_aborted_by_operator"


def _register_image_for_current_mission():
    """Increment image counter for current mission."""
    global CURRENT_MISSION_ID, MISSIONS
    if not CURRENT_MISSION_ID:
        _start_new_mission()
    mission = MISSIONS.get(CURRENT_MISSION_ID)
    if mission is None:
        return
    mission["images_count"] += 1
    if mission["status"] == "pending":
        mission["status"] = "running"


def _mission_summary(mission: dict) -> dict:
    return {
        "id": mission["id"],
        "started_at": mission["started_at"],
        "ended_at": mission["ended_at"],
        "status": mission["status"],
        "waypoints_reached": sorted(list(mission["waypoints_reached"])),
        "waypoints_unreachable": sorted(list(mission["waypoints_unreachable"])),
        "images_count": mission["images_count"],
    }


# ============================================================
# HTML / JS DASHBOARD
# ============================================================

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Strawberry Bot Command</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      :root {
        --bg-body: #020617;
        --bg-panel: #0f172a;
        --border: #1e293b;
        --accent: #38bdf8;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --success: #22c55e;
        --danger: #ef4444;
      }
      * { box-sizing: border-box; margin: 0; padding: 0; }
      
      body {
        font-family: system-ui, -apple-system, sans-serif;
        background: var(--bg-body); 
        color: var(--text-main);
        /* ENABLE SCROLLING ON BODY */
        min-height: 100vh;
        overflow-y: auto; 
        display: flex;
        flex-direction: column;
      }

      /* --- Top Navigation Bar (Fixed) --- */
      .navbar {
        position: fixed; top: 0; left: 0; right: 0; z-index: 50;
        height: 60px; background: var(--bg-panel); border-bottom: 1px solid var(--border);
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 20px;
      }
      .brand { font-size: 1.2rem; font-weight: 700; color: var(--accent); }
      
      .status-pill {
        font-size: 0.85rem; font-weight: 600; padding: 6px 16px; border-radius: 99px;
        background: #1e293b; border: 1px solid var(--border); color: var(--text-muted);
        display: flex; align-items: center; gap: 8px;
      }
      .dot { width: 8px; height: 8px; border-radius: 50%; background: #64748b; }
      .status-pill.running .dot { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
      .status-pill.running { color: var(--accent); border-color: rgba(56,189,248,0.3); }

      /* --- Main Layout --- */
      .container {
        display: flex;
        margin-top: 60px; /* Offset for fixed navbar */
        min-height: calc(100vh - 60px);
      }
      
      /* --- Sidebar (Fixed Left) --- */
      .sidebar {
        position: fixed; left: 0; top: 60px; bottom: 0; /* Lock to side */
        width: 300px; 
        background: #0b1120; border-right: 1px solid var(--border);
        display: flex; flex-direction: column;
        overflow-y: auto; /* Scroll sidebar independently if needed */
        z-index: 40;
      }
      .sidebar-section { padding: 20px; border-bottom: 1px solid var(--border); }
      .section-title { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px; margin-bottom: 12px; font-weight: 700; }

      /* Buttons */
      .btn-grid { display: grid; gap: 10px; }
      .btn {
        padding: 12px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;
        color: white; transition: all 0.2s; text-align: center;
      }
      .btn:disabled { opacity: 0.4; cursor: not-allowed; filter: grayscale(1); }
      .btn-start { background: linear-gradient(135deg, #22c55e, #16a34a); }
      .btn-abort { background: linear-gradient(135deg, #ef4444, #dc2626); }
      .btn-home { background: #334155; }

      /* Stats */
      .stat-row { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 8px; }
      .stat-val { font-family: monospace; color: var(--accent); }

      /* Mission List */
      .mission-list { list-style: none; padding: 0 20px 20px; }
      .mission-item {
        padding: 12px; background: #162033; border-radius: 8px; margin-bottom: 8px;
        border: 1px solid transparent; font-size: 0.85rem;
      }
      .m-header { display: flex; justify-content: space-between; margin-bottom: 4px; }
      .m-id { font-weight: bold; }
      .badge { font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; background: #334155; }

      /* --- Main Content (Scrollable Gallery) --- */
      .main-content {
        margin-left: 300px; /* Push content to right of sidebar */
        flex: 1;
        padding: 30px;
        background: var(--bg-body);
      }
      
      .gallery-grid {
        display: grid;
        /* Force minimum width of 300px per card to prevent squishing */
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 25px;
      }
      
      /* Image Cards */
      .card {
        background: #1e293b; border-radius: 12px; overflow: hidden;
        border: 1px solid var(--border); box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        display: flex; flex-direction: column;
        /* Ensure card never gets too small */
        min-width: 0; 
      }
      .card-header { 
        padding: 10px 15px; font-size: 0.85rem; color: var(--text-muted); 
        background: rgba(0,0,0,0.2); 
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; /* Prevent text wrap issues */
      }
      
      .img-row { display: flex; height: 200px; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
      .img-box { flex: 1; position: relative; cursor: pointer; overflow: hidden; }
      .img-box img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s; }
      .img-box:hover img { transform: scale(1.05); }
      .img-box:first-child { border-right: 1px solid var(--border); }
      
      .label {
        position: absolute; bottom: 8px; left: 8px; font-size: 0.65rem; font-weight: bold;
        padding: 3px 6px; border-radius: 4px; color: white; background: rgba(0,0,0,0.6); pointer-events: none;
      }
      .card-footer { padding: 12px 15px; font-size: 0.85rem; color: var(--text-muted); }
      .tag { color: var(--success); font-weight: 600; }

      /* Modal */
      .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 999; justify-content: center; align-items: center; backdrop-filter: blur(5px); }
      .modal img { max-width: 90vw; max-height: 90vh; border-radius: 8px; box-shadow: 0 0 50px rgba(0,0,0,0.5); }
      .close { position: absolute; top: 20px; right: 40px; color: white; font-size: 2rem; cursor: pointer; }

    </style>
  </head>
  <body>

    <div class="navbar">
      <div class="brand">🍓 StrawberryBot <span>v2.1</span></div>
      <div class="status-pill" id="global-status">
        <span class="dot"></span> <span id="status-text">CONNECTING...</span>
      </div>
    </div>

    <div class="container">
      
      <div class="sidebar">
        
        <div class="sidebar-section">
          <div class="section-title">Mission Control</div>
          <div class="btn-grid">
            <button id="btn-start" class="btn btn-start" onclick="startMission()">Start Mission</button>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
              <button id="btn-abort" class="btn btn-abort" onclick="abortMission()" disabled>Abort</button>
              <button id="btn-home" class="btn btn-home" onclick="returnHome()" disabled>Home</button>
            </div>
          </div>
        </div>

        <div class="sidebar-section">
          <div class="section-title">Live Telemetry</div>
          <div class="stat-row"><span>Status</span> <span class="stat-val" id="val-status">--</span></div>
          <div class="stat-row"><span>Waypoint</span> <span class="stat-val" id="val-wp">--</span></div>
          <div class="stat-row"><span>Photos</span> <span class="stat-val" id="val-imgs">0</span></div>
        </div>

        <div class="section-title" style="padding: 20px 20px 0;">Recent Missions</div>
        <ul class="mission-list" id="mission-list">
          <li class="mission-item" style="text-align:center; opacity:0.5;">Loading history...</li>
        </ul>
      </div>

      <div class="main-content">
        <div style="margin-bottom: 20px;">
          <h2 style="font-size:1.2rem; font-weight:600;">Live Inspection Feed</h2>
          <div style="font-size:0.8rem; color:var(--text-muted);">Real-time AI Analysis</div>
        </div>
        
        <div class="gallery-grid" id="gallery-grid">
          <div style="grid-column:1/-1; text-align:center; margin-top:50px; color:var(--text-muted);">
            Waiting for data...
          </div>
        </div>
      </div>
    </div>

    <div class="modal" id="imgModal" onclick="this.style.display='none'">
      <span class="close">&times;</span>
      <img id="modalImg" src="" />
    </div>

    <script>
      // API Calls
      async function startMission() {
        const btn = document.getElementById('btn-start');
        btn.innerText = "Starting...";
        btn.disabled = true;
        try { await fetch('/start_mission', {method:'POST'}); } 
        catch(e) { console.error(e); btn.innerText = "Start Mission"; btn.disabled = false; }
      }
      async function abortMission(){ try{ await fetch('/abort_mission',{method:'POST'});}catch(e){console.error(e);}}
      async function returnHome(){ try{ await fetch('/return_home',{method:'POST'});}catch(e){console.error(e);}}

      // Main Loop
      setInterval(() => { pollStatus(); pollMissions(); pollGallery(); }, 1500);

      async function pollStatus() {
        try {
          const r = await fetch('/last_status');
          const d = await r.json();
          const raw = (d.status || 'idle').toLowerCase();
          const run = raw.includes('moving') || raw.includes('started') || raw.includes('returning');
          
          document.getElementById('status-text').innerText = raw.replace(/_/g, ' ').toUpperCase();
          const pill = document.getElementById('global-status');
          pill.className = 'status-pill' + (run ? ' running' : '');
          
          document.getElementById('val-status').innerText = raw.includes('idle') ? 'IDLE' : 'ACTIVE';
          document.getElementById('val-wp').innerText = d.index !== undefined ? `WP-${d.index}` : '--';

          const btnStart = document.getElementById('btn-start');
          if(run) {
            btnStart.disabled = true; btnStart.innerText = "Running...";
            document.getElementById('btn-abort').disabled = false;
            document.getElementById('btn-home').disabled = true;
          } else {
            btnStart.disabled = false; btnStart.innerText = "Start Mission";
            document.getElementById('btn-abort').disabled = true;
            document.getElementById('btn-home').disabled = !raw.includes('abort');
          }
        } catch(e) {}
      }

      async function pollMissions() {
        try {
          const r = await fetch('/missions');
          const d = await r.json();
          const list = document.getElementById('mission-list');
          if(!d.missions || !d.missions.length) return;
          
          let html = '';
          d.missions.slice(0, 10).forEach(m => {
            const shortId = m.id.replace('mission_', '').substring(0,12);
            html += `<li class="mission-item">
              <div class="m-header"><span class="m-id">${shortId}...</span></div>
              <div style="font-size:0.75rem; color:#94a3b8">Photos: ${m.images_count} • ${m.status.toUpperCase()}</div>
            </li>`;
          });
          list.innerHTML = html;
        } catch(e) {}
      }

      async function pollGallery() {
        try {
          const r = await fetch('/latest_predictions');
          const d = await r.json();
          const grid = document.getElementById('gallery-grid');
          if(!d.photos || !d.photos.length) return;

          let html = '';
          d.photos.forEach(p => {
             const rawUrl = p.url.replace('pred_', '');
             const dets = (p.detections && p.detections.length) ? p.detections.join(', ') : 'No objects';
             const wp = p.label.split('·')[0] || 'WP';
             
             html += `
              <div class="card">
                <div class="card-header"><strong>${wp}</strong></div>
                <div class="img-row">
                  <div class="img-box" onclick="showModal('${rawUrl}')">
                    <img src="${rawUrl}" loading="lazy"><span class="label">RAW</span>
                  </div>
                  <div class="img-box" onclick="showModal('${p.url}')">
                    <img src="${p.url}" loading="lazy"><span class="label" style="background:#0ea5e9">AI</span>
                  </div>
                </div>
                <div class="card-footer">Found: <span class="tag">${dets}</span></div>
              </div>`;
          });
          grid.innerHTML = html;
          document.getElementById('val-imgs').innerText = d.photos.length;
        } catch(e) {}
      }

      function showModal(src) {
        document.getElementById('modalImg').src = src;
        document.getElementById('imgModal').style.display = 'flex';
      }
    </script>
  </body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/start_mission", methods=["POST"])
def start_mission():
    global MISSION_STATE, LAST_STATUS, LAST_TERMINAL_STATUS
    MISSION_STATE = "start"
    mission_id = _start_new_mission()
    LAST_STATUS.setdefault("status", "mission_idle")
    LAST_STATUS["mission_id"] = mission_id
    LAST_TERMINAL_STATUS = None
    print("[SERVER] Mission start requested via UI")
    return jsonify({"ok": True, "mission_id": mission_id})


@app.route("/mission_state", methods=["GET"])
def mission_state():
    global MISSION_STATE
    state = MISSION_STATE
    if MISSION_STATE == "start":
        MISSION_STATE = "idle"   # one-shot
    return jsonify({"mission_state": state})


@app.route("/abort_mission", methods=["POST"])
def abort_mission():
    global CONTROL_COMMAND
    CONTROL_COMMAND = "abort"
    print("[SERVER] Abort requested by operator")
    return jsonify({"ok": True, "command": "abort"})


@app.route("/return_home", methods=["POST"])
def return_home():
    global CONTROL_COMMAND
    CONTROL_COMMAND = "go_home"
    print("[SERVER] Return-home requested by operator")
    return jupytext({"ok": True, "command": "go_home"})


@app.route("/control_state", methods=["GET"])
def control_state():
    global CONTROL_COMMAND
    cmd = CONTROL_COMMAND
    CONTROL_COMMAND = "none"   # one-shot
    return jsonify({"command": cmd})


@app.route("/status_update", methods=["POST"])
def status_update():
    global LAST_STATUS, MISSION_STATE
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        payload = {"status": "bad_status_payload"}

    status = payload.get("status", "")

    if status == "mission_idle":
        MISSION_STATE = "idle"

    payload.setdefault("mission_state", MISSION_STATE)
    payload.setdefault("mission_id", CURRENT_MISSION_ID or "—")

    LAST_STATUS = payload
    print("[STATUS]", LAST_STATUS)

    _update_mission_on_status(payload)
    return jsonify({"ok": True})


@app.route("/last_status", methods=["GET"])
def last_status():
    global LAST_STATUS, LAST_TERMINAL_STATUS
    if not LAST_STATUS:
        return jsonify({
            "status": "no_status_yet",
            "mission_state": MISSION_STATE,
            "mission_id": CURRENT_MISSION_ID or "—",
            "terminal_status": LAST_TERMINAL_STATUS,
        })
    if "mission_state" not in LAST_STATUS:
        LAST_STATUS["mission_state"] = MISSION_STATE
    if "mission_id" not in LAST_STATUS:
        LAST_STATUS["mission_id"] = CURRENT_MISSION_ID or "—"
    LAST_STATUS["terminal_status"] = LAST_TERMINAL_STATUS
    return jsonify(LAST_STATUS)


@app.route("/missions", methods=["GET"])
def missions():
    items = [_mission_summary(m) for m in MISSIONS.values()]
    items_sorted = sorted(items, key=lambda x: x["started_at"] or "", reverse=True)
    return jsonify({"missions": items_sorted})


@app.route("/latest_photos", methods=["GET"])
def latest_photos():
    photos = []
    if not CURRENT_MISSION_ID:
        return jsonify({"photos": photos})

    mission_root = os.path.join(BASE_SAVE_DIR, CURRENT_MISSION_ID)
    if not os.path.isdir(mission_root):
        return jsonify({"photos": photos})

    for wp_name in sorted(os.listdir(mission_root)):
        wp_path = os.path.join(mission_root, wp_name)
        if not os.path.isdir(wp_path):
            continue
        files = [f for f in sorted(os.listdir(wp_path)) if f.startswith("img") and f.endswith(".jpg")]
        if not files:
            continue
        last_file = files[-1]
        url = f"/photo/{CURRENT_MISSION_ID}/{wp_name}/{last_file}"
        label = f"{wp_name} · {last_file}"
        photos.append({"url": url, "label": label})

    photos = photos[-8:]
    return jsonify({"photos": photos})


@app.route("/latest_predictions", methods=["GET"])
def latest_predictions():
    photos = []
    
    # CRITICAL CHANGE: Only look if we have an active/recent mission ID
    if not CURRENT_MISSION_ID:
        return jsonify({"photos": []})

    # Only look inside the folder for the CURRENT mission
    mission_root = os.path.join(BASE_SAVE_DIR, CURRENT_MISSION_ID)
    
    if not os.path.isdir(mission_root):
        return jsonify({"photos": []})

    # Scan through waypoints ONLY for this specific mission
    for wp_name in sorted(os.listdir(mission_root)):
        wp_path = os.path.join(mission_root, wp_name)
        if not os.path.isdir(wp_path):
            continue
            
        # Find prediction images in this waypoint
        files = [f for f in sorted(os.listdir(wp_path)) if f.startswith("pred_") and f.endswith(".jpg")]
        
        for file in files:
            # Extract image index to find the matching JSON file
            # e.g. pred_img3_2025... -> img3 -> 3
            try:
                img_part = file.split('_')[1] 
                image_index = img_part.replace('img', '')
            except:
                image_index = "unknown"

            # Load detections if they exist
            dets_file = f"pred_dets_{image_index}.json"
            dets_path = os.path.join(wp_path, dets_file)
            detections = []
            if os.path.isfile(dets_path):
                try:
                    with open(dets_path, "r") as f:
                        detections = json.load(f)
                except:
                    pass

            # Add to list
            url = f"/photo/{CURRENT_MISSION_ID}/{wp_name}/{file}"
            label = f"{wp_name} · {file.replace('pred_', '')}"
            photos.append({"url": url, "label": label, "detections": detections})

    # Return photos (newest at bottom usually, but the UI reverses it)
    return jsonify({"photos": photos})


@app.route("/photo/<mission_id>/<wp_folder>/<filename>")
def serve_photo(mission_id, wp_folder, filename):
    folder = os.path.join(BASE_SAVE_DIR, mission_id, wp_folder)
    full_path = os.path.join(folder, filename)
    print(f"[PHOTO_REQ] mission={mission_id} wp={wp_folder} file={filename}")
    print(f"           -> FS path: {full_path}")

    if not os.path.isfile(full_path):
        print("[PHOTO_REQ] FILE NOT FOUND")
        abort(404)

    return send_from_directory(folder, filename)


@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    waypoint_index = request.form.get("waypoint_index", "unknown")
    image_index = request.form.get("image_index", "0")
    file = request.files.get("image")

    if file is None:
        return jsonify({"status": "error", "message": "no image field"}), 400

    _register_image_for_current_mission()
    mission_id = CURRENT_MISSION_ID or "nomission"

    mission_folder = os.path.join(BASE_SAVE_DIR, mission_id)
    wp_folder_name = f"wp{waypoint_index}"
    wp_folder_path = os.path.join(mission_folder, wp_folder_name)
    os.makedirs(wp_folder_path, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"img{image_index}_{ts}.jpg"
    full_path = os.path.join(wp_folder_path, filename)

    file.save(full_path)
    print(f"[PHOTO_SAVE] {full_path}")

    # Run YOLO inference if model is loaded (on every photo, but UI shows only latest per waypoint)
    # Run YOLO inference if model is loaded
    if MODEL:
        try:
            # === CHANGED LINE: Added conf=0.5 to filter low confidence ===
            results = MODEL(full_path, conf=0.5)[0] 
            
            pred_filename = f"pred_{filename}"
            pred_path = os.path.join(wp_folder_path, pred_filename)
            
            # Save the image (only draws boxes > 50%)
            results.save(filename=pred_path)
            print(f"[YOLO_SAVE] {pred_path}")

            # Save detected classes as JSON (only includes boxes > 50%)
            detected = []
            if results.boxes:
                detected = list(set(results.names[int(cls)] for cls in results.boxes.cls))
            
            dets_path = os.path.join(wp_folder_path, f"pred_dets_{image_index}.json")
            with open(dets_path, "w") as f:
                json.dump(detected, f)
            print(f"[DETS_SAVE] {dets_path} - {detected}")
            
        except Exception as e:
            print(f"[YOLO_ERROR] {e}")
            
    return jsonify({
        "status": "ok",
        "filename": filename,
        "waypoint_folder": wp_folder_name,
        "mission_id": mission_id,
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"[SERVER] BASE_SAVE_DIR = {BASE_SAVE_DIR}")
    app.run(host="0.0.0.0", port=5000, debug=False)
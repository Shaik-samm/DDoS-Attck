"""
TSEDI Flask Backend  —  app.py
================================
Run:   python app.py
Open:  http://localhost:5000
"""
import os, json, uuid, datetime, warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = "tsedi2024"
CORS(app)

UPLOAD_DIR   = "uploads"
DATA_DIR     = "data"
USERS_FILE   = os.path.join(DATA_DIR, "users.json")
RESULTS_FILE = os.path.join(DATA_DIR, "results.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

_engine = None   # lazy-loaded ML engine

# ── helpers ───────────────────────────────────────────────────────────────
def rjson(path, default):
    try:
        if os.path.exists(path):
            with open(path) as f: return json.load(f)
    except Exception: pass
    return default

def wjson(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)

def get_engine():
    global _engine
    if _engine is None:
        from tsedi_engine import TSEDIEngine
        _engine = TSEDIEngine()
    return _engine

# ── pages ──────────────────────────────────────────────────────────────────
@app.route("/")
def page_index():     return render_template("index.html")

@app.route("/dashboard")
def page_dashboard(): return render_template("dashboard.html")

# ── auth ───────────────────────────────────────────────────────────────────
@app.route("/api/signup", methods=["POST"])
def signup():
    d     = request.get_json(force=True)
    users = rjson(USERS_FILE, [])
    phone = str(d.get("phone","")).strip()
    if any(u["phone"] == phone for u in users):
        return jsonify(ok=False, msg="Phone already registered"), 400
    users.append(dict(id=str(uuid.uuid4()),
                      firstName=d.get("firstName","").strip(),
                      lastName =d.get("lastName", "").strip(),
                      phone=phone, password=d.get("password",""),
                      devices=[], created=datetime.datetime.now().isoformat()))
    wjson(USERS_FILE, users)
    return jsonify(ok=True)

@app.route("/api/login", methods=["POST"])
def login():
    d     = request.get_json(force=True)
    users = rjson(USERS_FILE, [])
    user  = next((u for u in users
                  if u["phone"]    == str(d.get("phone",""))
                  and u["password"] == d.get("password","")), None)
    if not user:
        return jsonify(ok=False, msg="Invalid phone or password"), 401
    safe = {k: user[k] for k in ("id","firstName","lastName","phone","devices")}
    return jsonify(ok=True, user=safe)

# ── devices ────────────────────────────────────────────────────────────────
@app.route("/api/devices", methods=["POST"])
def add_device():
    d     = request.get_json(force=True)
    users = rjson(USERS_FILE, [])
    uid   = d.get("userId")
    for u in users:
        if u["id"] == uid:
            dev = dict(id=str(uuid.uuid4()),
                       name =d.get("name","Device"),
                       type =d.get("type","Other"),
                       ip   =d.get("ip","N/A"),
                       added=datetime.datetime.now().isoformat())
            u.setdefault("devices",[]).append(dev)
            wjson(USERS_FILE, users)
            return jsonify(ok=True, device=dev)
    return jsonify(ok=False, msg="User not found"), 404

@app.route("/api/devices/<uid>")
def get_devices(uid):
    users = rjson(USERS_FILE, [])
    user  = next((u for u in users if u["id"]==uid), None)
    return jsonify(user.get("devices",[]) if user else [])

# ── upload ─────────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(ok=False, msg="No file"), 400
    f     = request.files["file"]
    fname = str(uuid.uuid4()) + "_" + f.filename
    fpath = os.path.join(UPLOAD_DIR, fname)
    f.save(fpath)
    return jsonify(ok=True, filePath=fpath, fileName=f.filename)

# ── run analysis ───────────────────────────────────────────────────────────
@app.route("/api/run", methods=["POST"])
def run():
    d      = request.get_json(force=True)
    fpath  = d.get("filePath","")
    uid    = d.get("userId","")
    devid  = d.get("deviceId","")
    devnm  = d.get("deviceName","Device")

    if not os.path.exists(fpath):
        return jsonify(ok=False, msg="File not found"), 400

    eng    = get_engine()
    result = eng.analyse(fpath)

    result.update(id=str(uuid.uuid4()),
                  timestamp =datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S"),
                  deviceId  =devid,
                  deviceName=devnm,
                  userId    =uid,
                  fileName  =os.path.basename(fpath))

    if result.get("traffic") == "BLOCK":
        eng.self_evolve(fpath)
        result["evolved"] = True
    else:
        result["evolved"] = False

    all_r = rjson(RESULTS_FILE, [])
    all_r.insert(0, result)
    wjson(RESULTS_FILE, all_r)
    return jsonify(ok=True, result=result)

@app.route("/api/results/<uid>")
def get_results(uid):
    all_r = rjson(RESULTS_FILE, [])
    return jsonify([r for r in all_r if r.get("userId")==uid])

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  TSEDI  →  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)

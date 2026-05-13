"""
TrafficADS v3.0 - Complete Smart Route Planner Dashboard
Features:
  3 Routes: Fastest / Shortest / Safest with ETA, fuel, CO2
  Turn-by-turn navigation
  Weather impact on speed
  Active incidents with delay minutes
  Traffic heatmap (colored dots on map)
  Safety Score per route (0-100)
  Time-of-day traffic prediction (best time to leave)
  Road quality score
  Congestion trend chart (hourly)
  Nearby amenities panel
  Journey log / history
  Speed vs speed-limit comparison
  Mobile-friendly responsive layout
"""

from flask import Flask, render_template_string, jsonify, request
import pandas as pd, numpy as np, pickle, math, random, json
from datetime import datetime

app = Flask(__name__)

# ── JSON helper ────────────────────────────────────────────────────────────────
def to_python(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(i) for i in obj]
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, (np.bool_,)):     return bool(obj)
    if isinstance(obj, (np.ndarray,)):   return to_python(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)): return str(obj)
    return obj

def safe_json(data):
    return app.response_class(
        response=json.dumps(to_python(data)),
        status=200, mimetype="application/json"
    )

# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading datasets...")
traffic_df  = pd.read_csv("data/traffic_data.csv", parse_dates=["timestamp"])
pothole_df  = pd.read_csv("data/pothole_data.csv")
poi_df      = pd.read_csv("data/poi_data.csv")
weather_df  = pd.read_csv("data/weather_data.csv")
incident_df = pd.read_csv("data/incidents_data.csv")
road_df     = pd.read_csv("data/road_network.csv")

with open("models/congestion_model.pkl","rb") as f:
    saved     = pickle.load(f)
    ml_model  = saved["model"]
    scaler    = saved["scaler"]
    features  = saved["features"]
    model_name= saved.get("model_name","Random Forest")

traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
latest = traffic_df.sort_values("timestamp").drop_duplicates("road_id", keep="last").copy()

# Precompute hourly congestion pattern for charts
hourly_cong = traffic_df.groupby("hour")["congestion"].mean().round(3).to_dict()
hourly_speed= traffic_df.groupby("hour")["average_speed_kmph"].mean().round(1).to_dict()

print(f"  ✓ {len(traffic_df):,} traffic records | {len(poi_df)} POIs | {len(pothole_df)} hazards")

# ── Math helpers ───────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2)-float(lat1))
    dl = math.radians(float(lon2)-float(lon1))
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return round(2*R*math.asin(math.sqrt(max(0,min(1,a)))), 1)

def haversine_vec(lat1s, lon1s, lat2s, lon2s):
    R = 6371000
    lat1s=np.asarray(lat1s,dtype=float); lon1s=np.asarray(lon1s,dtype=float)
    lat2s=np.asarray(lat2s,dtype=float); lon2s=np.asarray(lon2s,dtype=float)
    p1=np.radians(lat1s); p2=np.radians(lat2s)
    dp=np.radians(lat2s-lat1s); dl=np.radians(lon2s-lon1s)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))

def seg_dist_vec(plats, plons, ax, ay, bx, by):
    plats=np.asarray(plats,dtype=float); plons=np.asarray(plons,dtype=float)
    ax,ay,bx,by = float(ax),float(ay),float(bx),float(by)
    abx,aby = bx-ax, by-ay
    t = np.clip(((plons-ax)*abx+(plats-ay)*aby)/(abx**2+aby**2+1e-9), 0, 1)
    return haversine_vec(plats, plons, ay+t*aby, ax+t*abx)

# ── Data helpers ───────────────────────────────────────────────────────────────
def get_weather():
    now = datetime.now()
    mask = weather_df["timestamp"].str.startswith(now.strftime("%Y-%m-%d %H"))
    row  = weather_df[mask].iloc[0] if mask.any() else weather_df.sample(1).iloc[0]
    return {
        "condition":    str(row["condition"]),
        "temp":         float(row["temperature_c"]),
        "humidity":     int(row["humidity_pct"]),
        "visibility":   float(row["visibility_km"]),
        "speed_factor": float(row["speed_factor"]),
    }

def get_corridor(src_lat, src_lon, dst_lat, dst_lon):
    pad = 0.06
    corr = latest[
        (latest.latitude  >= min(src_lat,dst_lat)-pad) &
        (latest.latitude  <= max(src_lat,dst_lat)+pad) &
        (latest.longitude >= min(src_lon,dst_lon)-pad) &
        (latest.longitude <= max(src_lon,dst_lon)+pad)
    ]
    if len(corr) < 5:
        lc = latest.copy()
        mid_lat = (src_lat+dst_lat)/2; mid_lon = (src_lon+dst_lon)/2
        lc["_d"] = haversine_vec(mid_lat,mid_lon,lc.latitude.values,lc.longitude.values)
        corr = lc.nsmallest(20,"_d")
    return corr

def best_departure_times():
    """Return top-3 best hours to depart based on historical congestion."""
    sorted_hours = sorted(hourly_cong.items(), key=lambda x: x[1])
    results = []
    for h, cong in sorted_hours[:3]:
        spd = hourly_speed.get(h, 45)
        label = f"{h:02d}:00"
        results.append({"hour":int(h), "label":label, "congestion_pct":int(cong*100), "avg_speed":float(spd)})
    return results

def road_quality_score(hazards_df):
    """0-100 score. Lower hazards + lower severity = higher score."""
    if len(hazards_df) == 0: return 100
    sev_weight = {"low":1, "medium":3, "high":7}
    total_weight = hazards_df["severity"].map(sev_weight).sum()
    score = max(0, 100 - int(total_weight * 2))
    return score

def safety_score(route_km, high_haz, incidents, cong_pct, weather_sf):
    """Composite safety score 0-100."""
    haz_penalty   = min(40, high_haz * 8)
    inc_penalty   = min(20, incidents * 5)
    cong_penalty  = min(15, cong_pct // 7)
    weather_pen   = int((1 - weather_sf) * 25)
    score = 100 - haz_penalty - inc_penalty - cong_penalty - weather_pen
    return max(0, min(100, score))

def compute_routes(src_lat,src_lon,dst_lat,dst_lon,corr,weather,rh,ri):
    dist_m   = haversine(src_lat,src_lon,dst_lat,dst_lon)
    dist_km  = round(dist_m/1000, 2)
    wf       = weather["speed_factor"]
    avg_spd  = float(corr.average_speed_kmph.mean()) if len(corr) else 45.0
    avg_lim  = float(corr.speed_limit_kmph.mean())   if len(corr) else 50.0
    cong_pct = int(corr.congestion.mean()*100)        if len(corr) else 30
    high_haz = int((rh.severity=="high").sum())
    act_inc  = len(ri)
    inc_del  = float(ri.delay_added_min.sum()) if len(ri) else 0

    def eta(km, spd, extra=0):
        return round((km/max(spd,5))*60 + extra, 1)

    def waypoints(n, jitter=0.006):
        pts = []
        for i in range(1, n+1):
            f = i/(n+1)
            pts.append({
                "lat": round(src_lat+(dst_lat-src_lat)*f + random.gauss(0,jitter), 5),
                "lon": round(src_lon+(dst_lon-src_lon)*f + random.gauss(0,jitter), 5),
            })
        return pts

    # Road names used
    roads = corr.road_name.unique().tolist()[:5] if len(corr) else ["Main Road","Ring Road","Bypass"]

    def steps(km, road_list):
        turns  = ["Head north on","Turn right onto","Turn left onto","Continue on","Merge onto","Take the ramp onto"]
        segs   = []
        seg_km = round(km/max(len(road_list),1), 1)
        for i,r in enumerate(road_list):
            segs.append({"step":i+1,"instruction":f"{turns[i%len(turns)]} {r}","distance_km":seg_km,"road":r})
        segs.append({"step":len(segs)+1,"instruction":"Arrive at your destination","distance_km":0.0,"road":""})
        return segs

    r1_km  = round(dist_km*1.22, 2)
    r1_spd = round(min(avg_lim+5, avg_spd*1.18*wf), 1)
    r1_eta = eta(r1_km, r1_spd, extra=inc_del*0.2)
    r1_haz = max(0, high_haz-2)
    r1_ss  = safety_score(r1_km, r1_haz, act_inc, max(0,cong_pct-15), wf)

    r2_km  = round(dist_km*1.10, 2)
    r2_spd = round(avg_spd*wf, 1)
    r2_eta = eta(r2_km, r2_spd, extra=inc_del*0.5)
    r2_haz = high_haz
    r2_ss  = safety_score(r2_km, r2_haz, act_inc, cong_pct, wf)

    r3_km  = round(dist_km*1.48, 2)
    r3_spd = round(avg_spd*0.88*wf, 1)
    r3_eta = eta(r3_km, r3_spd, extra=0)
    r3_haz = 0
    r3_ss  = safety_score(r3_km, 0, 0, max(0,cong_pct-25), wf)

    return {
        "fastest": {
            "label":"⚡ Fastest", "color":"#F59E0B",
            "distance_km":r1_km, "eta_min":r1_eta, "avg_speed":r1_spd,
            "high_hazards":r1_haz, "fuel_litres":round(r1_km*0.08,2),
            "co2_grams":int(r1_km*185), "congestion_pct":max(0,cong_pct-15),
            "safety_score":r1_ss,
            "via":"Outer Ring Road / Highway",
            "waypoints":waypoints(3, 0.004), "steps":steps(r1_km, roads),
        },
        "shortest": {
            "label":"📍 Shortest", "color":"#3B82F6",
            "distance_km":r2_km, "eta_min":r2_eta, "avg_speed":r2_spd,
            "high_hazards":r2_haz, "fuel_litres":round(r2_km*0.09,2),
            "co2_grams":int(r2_km*210), "congestion_pct":cong_pct,
            "safety_score":r2_ss,
            "via":"Anna Salai / Inner Ring Road",
            "waypoints":waypoints(2, 0.003), "steps":steps(r2_km, roads),
        },
        "safest": {
            "label":"🛡️ Safest", "color":"#10B981",
            "distance_km":r3_km, "eta_min":r3_eta, "avg_speed":r3_spd,
            "high_hazards":r3_haz, "fuel_litres":round(r3_km*0.075,2),
            "co2_grams":int(r3_km*170), "congestion_pct":max(0,cong_pct-30),
            "safety_score":r3_ss,
            "via":"ECR / Bypass / Avoid Hotspots",
            "waypoints":waypoints(5, 0.007), "steps":steps(r3_km, roads),
        },
    }

# ── Landmarks ──────────────────────────────────────────────────────────────────
LANDMARKS = [
    {"name":"Central Railway Station","lat":13.0827,"lon":80.2707},
    {"name":"Chennai Airport","lat":12.9941,"lon":80.1709},
    {"name":"Marina Beach","lat":13.0500,"lon":80.2824},
    {"name":"T Nagar","lat":13.0418,"lon":80.2341},
    {"name":"Anna Nagar","lat":13.0850,"lon":80.2101},
    {"name":"Adyar","lat":13.0012,"lon":80.2565},
    {"name":"Tambaram","lat":12.9249,"lon":80.1000},
    {"name":"Porur","lat":13.0368,"lon":80.1572},
    {"name":"Velachery","lat":12.9815,"lon":80.2209},
    {"name":"Guindy","lat":13.0067,"lon":80.2206},
    {"name":"Egmore","lat":13.0732,"lon":80.2609},
    {"name":"Perambur","lat":13.1189,"lon":80.2359},
    {"name":"Chromepet","lat":12.9516,"lon":80.1462},
    {"name":"Sholinganallur","lat":12.9010,"lon":80.2279},
    {"name":"OMR IT Corridor","lat":12.9500,"lon":80.2300},
    {"name":"Koyambedu Bus Stand","lat":13.0700,"lon":80.1950},
    {"name":"Vadapalani","lat":13.0520,"lon":80.2120},
    {"name":"Thiruvanmiyur","lat":12.9830,"lon":80.2590},
    {"name":"Nungambakkam","lat":13.0580,"lon":80.2420},
    {"name":"Pallavaram","lat":12.9674,"lon":80.1492},
]

# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>TrafficADS v3 — Smart Route Planner</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0D1117;--panel:#111827;--card:#1E293B;--bdr:#334155;
  --teal:#0D9488;--teal2:#14B8A6;--navy:#0F2044;
  --green:#10B981;--red:#EF4444;--yellow:#F59E0B;--blue:#3B82F6;--purple:#8B5CF6;
  --white:#F1F5F9;--gray:#64748B;--lgray:#94A3B8;
}
body{font-family:'Segoe UI',sans-serif;background:var(--bg);color:var(--white);display:flex;flex-direction:column;height:100vh;overflow:hidden}

/* HEADER */
header{background:linear-gradient(135deg,var(--navy),#1a3a6e);padding:10px 20px;display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid var(--teal);flex-shrink:0;z-index:1000}
header h1{font-size:1.05rem;color:#93C5FD;font-weight:700}
header p{font-size:.68rem;color:var(--lgray);margin-top:1px}
.live-badge{background:#14532d;color:#4ADE80;padding:3px 10px;border-radius:20px;font-size:.68rem;font-weight:700;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.7}}

/* LAYOUT */
.main{display:flex;flex:1;overflow:hidden}
.left{width:370px;flex-shrink:0;background:var(--panel);border-right:1px solid var(--bdr);display:flex;flex-direction:column;overflow:hidden}
#map{flex:1;position:relative}

/* TABS */
.top-tabs{display:flex;border-bottom:1px solid var(--bdr);flex-shrink:0}
.top-tab{flex:1;padding:8px 4px;font-size:.72rem;color:var(--gray);cursor:pointer;text-align:center;border-bottom:2px solid transparent;transition:.2s}
.top-tab.active{color:var(--teal2);border-bottom-color:var(--teal2);font-weight:600}

.tab-content{display:none;flex-direction:column;overflow:hidden;flex:1}
.tab-content.active{display:flex}

/* SEARCH PANEL */
.search-box{padding:12px;border-bottom:1px solid var(--bdr);flex-shrink:0}
.section-label{font-size:.68rem;color:var(--teal);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;font-weight:700}
.loc-row{display:flex;align-items:center;gap:7px;margin-bottom:7px}
.dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.dot.src{background:var(--green)}.dot.dst{background:var(--red)}
select.ls{flex:1;background:var(--card);border:1px solid var(--bdr);border-radius:7px;color:var(--white);padding:7px 10px;font-size:.78rem}
select.ls:focus{outline:none;border-color:var(--teal)}
.hint{font-size:.67rem;color:var(--gray);text-align:center;padding:3px 0 6px}
.route-tabs{display:flex;gap:5px;margin-bottom:8px}
.rtab{flex:1;padding:6px 0;border:1px solid var(--bdr);border-radius:7px;font-size:.71rem;cursor:pointer;background:var(--card);color:var(--lgray);text-align:center;transition:.2s}
.rtab.active{background:var(--teal);border-color:var(--teal);color:#fff;font-weight:700}
.rtab:hover:not(.active){border-color:var(--teal2);color:var(--teal2)}
.btn-main{width:100%;padding:9px;background:linear-gradient(135deg,var(--teal),#0891B2);border:none;border-radius:8px;color:#fff;font-size:.85rem;font-weight:700;cursor:pointer;transition:.2s}
.btn-main:hover:not(:disabled){filter:brightness(1.15)}
.btn-main:disabled{background:var(--card);color:var(--gray);cursor:not-allowed}
.btn-sec{width:100%;margin-top:5px;padding:6px;background:transparent;border:1px solid var(--bdr);border-radius:8px;color:var(--lgray);font-size:.74rem;cursor:pointer}
.btn-sec:hover{border-color:var(--red);color:var(--red)}

/* RESULTS PANEL */
.results{overflow-y:auto;padding:10px;flex:1}
.results::-webkit-scrollbar{width:3px}.results::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:2px}

/* CARDS */
.rcard{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:11px;margin-bottom:8px}
.rcard h4{font-size:.72rem;color:var(--teal2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:9px;display:flex;align-items:center;gap:5px}
.sr{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid #1a2235;font-size:.78rem}
.sr:last-child{border-bottom:none}
.sv{font-weight:700}
.g{color:var(--green)}.r{color:var(--red)}.y{color:var(--yellow)}.b{color:var(--blue)}.p{color:var(--purple)}.t{color:var(--teal2)}

/* ROUTE CARDS */
.route-card{border:2px solid var(--bdr);border-radius:9px;padding:10px;margin-bottom:6px;cursor:pointer;transition:.2s;background:var(--bg)}
.route-card:hover,.route-card.sel{border-color:var(--teal);background:#0a1628}
.route-card h5{font-size:.85rem;font-weight:700;margin-bottom:5px}
.route-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin:6px 0}
.rg{text-align:center;padding:4px 2px;background:var(--panel);border-radius:5px}
.rg-v{font-size:.85rem;font-weight:700}.rg-l{font-size:.58rem;color:var(--gray);margin-top:1px}
.route-pills{display:flex;gap:6px;flex-wrap:wrap;margin-top:4px}
.pill{font-size:.65rem;padding:2px 7px;border-radius:10px;background:var(--panel)}

/* SAFETY METER */
.safety-bar{height:6px;border-radius:3px;background:var(--bdr);margin:4px 0 2px;overflow:hidden}
.safety-fill{height:100%;border-radius:3px;transition:.4s}

/* STEPS */
.step{display:flex;gap:7px;padding:4px 0;border-bottom:1px solid #1a2235;font-size:.75rem;align-items:flex-start}
.step:last-child{border-bottom:none}
.step-n{background:var(--teal);color:#fff;border-radius:50%;width:18px;height:18px;display:flex;align-items:center;justify-content:center;font-size:.62rem;font-weight:700;flex-shrink:0;margin-top:1px}

/* POI / HAZARD */
.poi-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #1a2235;font-size:.76rem}
.poi-row:last-child{border-bottom:none}
.hz-row{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid #1a2235;font-size:.75rem}
.hz-row:last-child{border-bottom:none}
.tag{font-size:.62rem;padding:2px 7px;border-radius:9px;white-space:nowrap}
.t-hospital{background:#7f1d1d;color:#fca5a5}.t-fuel_station{background:#78350f;color:#fcd34d}
.t-shop{background:#1e3a5f;color:#93c5fd}.t-restaurant{background:#14532d;color:#86efac}
.t-atm{background:#3b0764;color:#d8b4fe}.t-pharmacy{background:#7c2d12;color:#fdba74}
.t-school{background:#164e63;color:#67e8f9}.t-parking{background:#1e293b;color:#94a3b8}
.t-bank{background:#1e3a5f;color:#bfdbfe}.t-metro_station{background:#1e1b4b;color:#a5b4fc}
.t-ev_station{background:#14532d;color:#4ade80}.t-police_station{background:#1c1917;color:#d6d3d1}
.sev-high{background:#7f1d1d;color:#fca5a5;padding:2px 6px;border-radius:8px;font-size:.62rem}
.sev-medium{background:#78350f;color:#fcd34d;padding:2px 6px;border-radius:8px;font-size:.62rem}
.sev-low{background:#14532d;color:#86efac;padding:2px 6px;border-radius:8px;font-size:.62rem}

/* WEATHER BAR */
.w-bar{display:flex;align-items:center;gap:8px;padding:7px 10px;background:#0f1f3d;border-radius:8px;margin-bottom:8px;font-size:.75rem;border:1px solid #1e3a5f}
.w-icon{font-size:1.4rem}

/* HEATMAP LEGEND */
.legend-box{position:absolute;bottom:20px;right:10px;background:rgba(13,17,23,.92);border:1px solid var(--bdr);border-radius:8px;padding:8px 12px;font-size:.7rem;z-index:1000}
.legend-item{display:flex;align-items:center;gap:6px;margin:2px 0}
.legend-dot{width:10px;height:10px;border-radius:50%}

/* CHART */
.chart-wrap{overflow-x:auto;padding:4px 0}
canvas{width:100%!important}

/* DEPARTURE TABLE */
.dep-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #1a2235;font-size:.76rem}
.dep-row:last-child{border-bottom:none}

/* JOURNEY LOG */
.log-item{background:var(--bg);border-radius:6px;padding:7px 9px;margin-bottom:5px;font-size:.74rem;border:1px solid var(--bdr)}

/* SPINNER */
.spinner{display:inline-block;width:20px;height:20px;border:3px solid var(--bdr);border-top-color:var(--teal);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loading{text-align:center;padding:30px;color:var(--gray)}
.placeholder{text-align:center;padding:40px 16px;color:var(--bdr)}
.placeholder .icon{font-size:2.4rem;margin-bottom:8px}
</style>
</head>
<body>

<header>
  <div>
    <h1>🚦 TrafficADS v3.0 — Smart Route Planner</h1>
    <p>GPU-Accelerated RAPIDS ADS · 50K Records · 800 POIs · 1200 Hazards · 6 Datasets · Chennai</p>
  </div>
  <span class="live-badge">● LIVE</span>
</header>

<div class="main">
  <!-- LEFT PANEL -->
  <div class="left">
    <div class="top-tabs">
      <div class="top-tab active" onclick="switchTab('route')"   id="tt-route">🗺️ Route</div>
      <div class="top-tab"        onclick="switchTab('traffic')" id="tt-traffic">📊 Traffic</div>
      <div class="top-tab"        onclick="switchTab('history')" id="tt-history">📋 History</div>
    </div>

    <!-- ROUTE TAB -->
    <div class="tab-content active" id="tab-route">
      <div class="search-box">
        <div class="section-label">Select Route</div>
        <div class="loc-row">
          <div class="dot src"></div>
          <select class="ls" id="src-sel"><option value="">— Source —</option></select>
        </div>
        <div class="loc-row">
          <div class="dot dst"></div>
          <select class="ls" id="dst-sel"><option value="">— Destination —</option></select>
        </div>
        <div class="hint" id="map-hint">Or click directly on the map to set points</div>
        <div class="route-tabs">
          <div class="rtab active" onclick="setMode('fastest')"  id="rtab-fastest">⚡ Fastest</div>
          <div class="rtab"        onclick="setMode('shortest')" id="rtab-shortest">📍 Shortest</div>
          <div class="rtab"        onclick="setMode('safest')"   id="rtab-safest">🛡️ Safest</div>
        </div>
        <button class="btn-main" id="btn-analyze" disabled onclick="analyzeRoute()">⚡ Analyze Route</button>
        <button class="btn-sec"  onclick="clearAll()">✕ Clear Route</button>
      </div>
      <div class="results" id="results-panel">
        <div class="placeholder"><div class="icon">🗺️</div><div>Pick source &amp; destination<br>then click Analyze Route</div></div>
      </div>
    </div>

    <!-- TRAFFIC TAB -->
    <div class="tab-content" id="tab-traffic">
      <div class="results" id="traffic-panel">
        <div class="loading"><div class="spinner"></div><br>Loading traffic data…</div>
      </div>
    </div>

    <!-- HISTORY TAB -->
    <div class="tab-content" id="tab-history">
      <div class="results" id="history-panel">
        <div class="placeholder"><div class="icon">📋</div><div>Your journey history<br>will appear here</div></div>
      </div>
    </div>
  </div>

  <!-- MAP -->
  <div id="map">
    <div class="legend-box" id="map-legend" style="display:none">
      <b style="font-size:.72rem;color:#94a3b8">MAP LEGEND</b>
      <div class="legend-item"><div class="legend-dot" style="background:#10B981"></div> Free flow</div>
      <div class="legend-item"><div class="legend-dot" style="background:#F59E0B"></div> Moderate</div>
      <div class="legend-item"><div class="legend-dot" style="background:#EF4444"></div> Heavy</div>
      <div class="legend-item"><div class="legend-dot" style="background:#8B5CF6"></div> Incident</div>
      <div class="legend-item"><div class="legend-dot" style="background:#F97316"></div> Hazard (High)</div>
    </div>
  </div>
</div>

<script>
const LM = {{ landmarks|tojson }};
const HOURLY_CONG  = {{ hourly_cong|tojson }};
const HOURLY_SPEED = {{ hourly_speed|tojson }};

// ── Map setup ─────────────────────────────────────────────────────────────────
const map = L.map('map',{center:[13.05,80.22],zoom:11});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap'}).addTo(map);

let srcLL=null,dstLL=null,srcM=null,dstM=null;
let routeLayers=[],hazardL=null,poiL=null,incL=null,heatL=null;
let selectedMode='fastest', lastData=null;
let journeyLog=[];

// ── Dropdowns ─────────────────────────────────────────────────────────────────
const srcSel=document.getElementById('src-sel');
const dstSel=document.getElementById('dst-sel');
LM.forEach((lm,i)=>{
  [srcSel,dstSel].forEach(sel=>{
    const o=document.createElement('option'); o.value=i; o.textContent=lm.name; sel.appendChild(o);
  });
});
srcSel.onchange=()=>{ if(srcSel.value!==''){const l=LM[+srcSel.value];setSource(l.lat,l.lon,l.name);}};
dstSel.onchange=()=>{ if(dstSel.value!==''){const l=LM[+dstSel.value];setDest(l.lat,l.lon,l.name);}};
map.on('click',e=>{
  const{lat,lng}=e.latlng;
  if(!srcLL) setSource(lat,lng,`${lat.toFixed(4)}, ${lng.toFixed(4)}`);
  else if(!dstLL) setDest(lat,lng,`${lat.toFixed(4)}, ${lng.toFixed(4)}`);
});

// ── Markers ───────────────────────────────────────────────────────────────────
function mkIcon(bg, text){
  return L.divIcon({className:'',iconAnchor:[0,10],
    html:`<div style="background:${bg};color:#fff;padding:3px 9px;border-radius:16px;font-size:10px;font-weight:700;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.7)">${text}</div>`});
}
function setSource(lat,lon,label){
  srcLL={lat,lon}; if(srcM) map.removeLayer(srcM);
  srcM=L.marker([lat,lon],{icon:mkIcon('#10B981','🟢 '+label)}).addTo(map); updateHint();
}
function setDest(lat,lon,label){
  dstLL={lat,lon}; if(dstM) map.removeLayer(dstM);
  dstM=L.marker([lat,lon],{icon:mkIcon('#EF4444','🔴 '+label)}).addTo(map);
  if(srcLL) map.fitBounds([[srcLL.lat,srcLL.lon],[lat,lon]],{padding:[60,60]});
  updateHint();
}
function updateHint(){
  const el=document.getElementById('map-hint');
  const btn=document.getElementById('btn-analyze');
  if(!srcLL){el.textContent='Click map or use dropdown to set Source';btn.disabled=true;}
  else if(!dstLL){el.textContent='✅ Source set — now pick Destination';btn.disabled=true;}
  else{el.textContent='✅ Both set — choose route type & analyze';btn.disabled=false;}
}
function setMode(m){
  selectedMode=m;
  ['fastest','shortest','safest'].forEach(k=>{
    document.getElementById('rtab-'+k).className='rtab'+(k===m?' active':'');
  });
  if(lastData) renderResults(lastData);
  if(lastData) drawMap(lastData);
}
function clearAll(){
  srcLL=dstLL=null; [srcM,dstM].forEach(m=>m&&map.removeLayer(m)); srcM=dstM=null;
  clearMapLayers(); srcSel.value=''; dstSel.value=''; lastData=null;
  document.getElementById('results-panel').innerHTML=`<div class="placeholder"><div class="icon">🗺️</div><div>Pick source &amp; destination</div></div>`;
  document.getElementById('map-legend').style.display='none';
  updateHint();
}
function clearMapLayers(){
  routeLayers.forEach(l=>map.removeLayer(l)); routeLayers=[];
  [hazardL,poiL,incL,heatL].forEach(l=>l&&map.removeLayer(l)); hazardL=poiL=incL=heatL=null;
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name){
  ['route','traffic','history'].forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('active',t===name);
    document.getElementById('tt-'+t).classList.toggle('active',t===name);
  });
  if(name==='traffic') loadTrafficPanel();
}

// ── Analyze ───────────────────────────────────────────────────────────────────
async function analyzeRoute(){
  document.getElementById('results-panel').innerHTML=`<div class="loading"><div class="spinner"></div><br><br>Analyzing route…<br><small style="color:#475569">Processing 50K traffic records + weather + incidents…</small></div>`;
  document.getElementById('btn-analyze').disabled=true;
  try{
    const res=await fetch('/analyze',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({src_lat:srcLL.lat,src_lon:srcLL.lon,dst_lat:dstLL.lat,dst_lon:dstLL.lon})
    });
    lastData=await res.json();
    renderResults(lastData);
    drawMap(lastData);
    document.getElementById('map-legend').style.display='block';
    saveToHistory(lastData);
  }catch(e){
    document.getElementById('results-panel').innerHTML=`<div class="loading" style="color:#EF4444">❌ Error: ${e.message}</div>`;
  }
  document.getElementById('btn-analyze').disabled=false;
}

// ── Draw map ──────────────────────────────────────────────────────────────────
const catEmoji={hospital:'🏥',fuel_station:'⛽',shop:'🛒',restaurant:'🍽️',atm:'🏧',
  pharmacy:'💊',school:'🏫',parking:'🅿️',bank:'🏦',metro_station:'🚇',ev_station:'⚡',police_station:'🚔'};
const rColors={fastest:'#F59E0B',shortest:'#3B82F6',safest:'#10B981'};

function drawMap(d){
  clearMapLayers();
  // Traffic heatmap dots
  heatL=L.layerGroup();
  if(d.heatmap) d.heatmap.forEach(p=>{
    const c=p.level==='heavy'?'#EF4444':p.level==='moderate'?'#F59E0B':'#10B981';
    L.circleMarker([p.lat,p.lon],{radius:5,fillColor:c,color:'transparent',fillOpacity:.6}).addTo(heatL);
  });
  heatL.addTo(map);

  // All 3 routes (faded lines)
  Object.entries(d.routes).forEach(([key,r])=>{
    const isSel=key===selectedMode;
    const pts=[[srcLL.lat,srcLL.lon],...r.waypoints.map(w=>[w.lat,w.lon]),[dstLL.lat,dstLL.lon]];
    const l=L.polyline(pts,{color:rColors[key],weight:isSel?5:2,opacity:isSel?.9:.25,dashArray:isSel?null:'8,5'}).addTo(map);
    if(isSel) l.bindPopup(`<b>${r.label}</b><br>${r.eta_min} min · ${r.distance_km} km`);
    routeLayers.push(l);
  });

  // Hazards
  hazardL=L.layerGroup();
  d.hazards.list.forEach(h=>{
    const c=h.severity==='high'?'#F97316':h.severity==='medium'?'#F59E0B':'#10B981';
    L.circleMarker([h.latitude,h.longitude],{radius:h.severity==='high'?9:6,fillColor:c,color:'#0D1117',weight:1,fillOpacity:.88})
      .bindPopup(`<b>${h.hazard_type.replace(/_/g,' ')}</b><br>Severity: <b>${h.severity}</b><br>Confidence: ${(h.confidence_score*100).toFixed(0)}%${h.depth_cm?'<br>Depth: '+h.depth_cm+'cm':''}<br>Repair cost: ₹${h.repair_cost||'N/A'}`)
      .addTo(hazardL);
  });
  hazardL.addTo(map);

  // POIs
  poiL=L.layerGroup();
  d.pois.list.forEach(p=>{
    L.marker([p.latitude,p.longitude],{icon:L.divIcon({className:'',iconAnchor:[10,10],
      html:`<div style="font-size:15px;filter:drop-shadow(0 1px 3px rgba(0,0,0,.9))">${catEmoji[p.category]||'📍'}</div>`})})
      .bindPopup(`<b>${p.name}</b><br>${p.category.replace(/_/g,' ')}<br>⭐ ${p.rating}<br>${p.open_now?'🟢 Open':'🔴 Closed'} · ${p.open_hours||''}<br><small>${Math.round(p.dist_m)}m from route</small>`)
      .addTo(poiL);
  });
  poiL.addTo(map);

  // Incidents
  incL=L.layerGroup();
  d.incidents.list.forEach(i=>{
    L.circleMarker([i.latitude,i.longitude],{radius:9,fillColor:'#8B5CF6',color:'#1e1b4b',weight:2,fillOpacity:.9})
      .bindPopup(`<b>⚠ ${i.incident_type.replace(/_/g,' ')}</b><br>Severity: ${i.severity}<br>+${i.delay_added_min} min delay<br>${i.road_name}`)
      .addTo(incL);
  });
  incL.addTo(map);
}

// ── Render results ────────────────────────────────────────────────────────────
function safetyColor(s){ return s>=80?'#10B981':s>=55?'#F59E0B':'#EF4444'; }

function renderResults(d){
  const r=d.routes[selectedMode];
  const wIcon={'clear':'☀️','rain':'🌧️','drizzle':'🌦️','fog':'🌫️','heavy_rain':'⛈️'}[d.weather.condition]||'🌤️';
  const wSlow = d.weather.speed_factor<0.85;
  const sc=safetyColor(r.safety_score);

  const routeCards=['fastest','shortest','safest'].map(k=>{
    const rr=d.routes[k]; const isSel=k===selectedMode; const sc2=safetyColor(rr.safety_score);
    return `<div class="route-card${isSel?' sel':''}" onclick="setMode('${k}')">
      <h5 style="color:${rColors[k]}">${rr.label}</h5>
      <div style="font-size:.68rem;color:var(--gray);margin-bottom:3px">via ${rr.via}</div>
      <div class="route-grid">
        <div class="rg"><div class="rg-v" style="color:${rColors[k]}">${rr.eta_min}<span style="font-size:.6rem">min</span></div><div class="rg-l">ETA</div></div>
        <div class="rg"><div class="rg-v">${rr.distance_km}<span style="font-size:.6rem">km</span></div><div class="rg-l">Distance</div></div>
        <div class="rg"><div class="rg-v">${rr.avg_speed}<span style="font-size:.6rem">km/h</span></div><div class="rg-l">Avg Speed</div></div>
        <div class="rg"><div class="rg-v" style="color:${sc2}">${rr.safety_score}</div><div class="rg-l">Safety</div></div>
      </div>
      <div class="safety-bar"><div class="safety-fill" style="width:${rr.safety_score}%;background:${sc2}"></div></div>
      <div class="route-pills">
        <span class="pill" style="color:#F59E0B">⛽ ${rr.fuel_litres}L</span>
        <span class="pill" style="color:#64748B">💨 ${rr.co2_grams}g CO₂</span>
        <span class="pill" style="color:${rr.high_hazards>0?'#EF4444':'#10B981'}">🕳️ ${rr.high_hazards} high haz</span>
        <span class="pill" style="color:${rr.congestion_pct>50?'#EF4444':'#10B981'}">${rr.congestion_pct}% cong</span>
      </div>
    </div>`;
  }).join('');

  const steps=r.steps.slice(0,7).map(s=>`
    <div class="step"><div class="step-n">${s.step}</div>
    <div>${s.instruction}${s.distance_km>0?` <span style="color:var(--gray)">(${s.distance_km} km)</span>`:''}</div></div>`).join('');

  const hzHtml=d.hazards.list.slice(0,6).map(h=>`
    <div class="hz-row">
      <div><b>${h.hazard_type.replace(/_/g,' ')}</b>${h.depth_cm?`<span style="color:var(--gray);font-size:.65rem"> · ${h.depth_cm}cm</span>`:''}<br>
      <span style="color:var(--gray);font-size:.65rem">Repair: ₹${h.repair_cost||'—'}</span></div>
      <span class="sev-${h.severity}">${h.severity.toUpperCase()}</span>
    </div>`).join('')||'<div style="color:var(--gray);font-size:.75rem;padding:4px">✅ No hazards on this route</div>';

  const poiHtml=d.pois.list.slice(0,10).map(p=>`
    <div class="poi-row">
      <div><div style="font-weight:600">${p.name}</div>
      <div style="font-size:.65rem;color:var(--gray)">${Math.round(p.dist_m)}m · ⭐${p.rating} · ${p.open_now?'<span style="color:#10B981">Open</span>':'<span style="color:#EF4444">Closed</span>'}</div></div>
      <span class="tag t-${p.category}">${p.category.replace(/_/g,' ')}</span>
    </div>`).join('');

  const incHtml=d.incidents.list.slice(0,4).map(i=>`
    <div class="hz-row">
      <div><b>${i.incident_type.replace(/_/g,' ')}</b><br><span style="color:var(--gray);font-size:.65rem">${i.road_name||'Unknown road'}</span></div>
      <span class="sev-${i.severity}">+${i.delay_added_min}min</span>
    </div>`).join('')||'<div style="color:var(--gray);font-size:.75rem;padding:4px">✅ No active incidents</div>';

  const depRows=d.best_departure.map(dep=>{
    const best=dep===d.best_departure[0];
    return `<div class="dep-row">
      <div><b>${dep.label}</b>${best?' <span style="color:#10B981;font-size:.65rem">✅ Best</span>':''}</div>
      <div style="text-align:right"><span class="${dep.congestion_pct<30?'g':dep.congestion_pct<60?'y':'r'}">${dep.congestion_pct}% cong</span><br>
      <span style="color:var(--gray);font-size:.65rem">${dep.avg_speed} km/h</span></div>
    </div>`;
  }).join('');

  document.getElementById('results-panel').innerHTML=`
    <div class="w-bar">
      <div class="w-icon">${wIcon}</div>
      <div><b>${d.weather.condition.toUpperCase()}</b> &nbsp;${d.weather.temp}°C &nbsp;💧${d.weather.humidity}% &nbsp;👁 ${d.weather.visibility}km</div>
      <div style="margin-left:auto;font-size:.68rem">${wSlow?`<span class="r">⚠ ${Math.round((1-d.weather.speed_factor)*100)}% speed reduction</span>`:'<span class="g">✅ Good conditions</span>'}</div>
    </div>

    <div class="rcard"><h4>🛣️ Choose Route</h4>${routeCards}</div>

    <div class="rcard">
      <h4>🧭 Turn-by-Turn — ${r.label}</h4>
      ${steps}
    </div>

    <div class="rcard">
      <h4>📊 Corridor Traffic</h4>
      <div class="sr"><span>Roads Monitored</span><span class="sv b">${d.traffic.roads_monitored}</span></div>
      <div class="sr"><span>Congested Roads</span><span class="sv r">${d.traffic.congested_roads}</span></div>
      <div class="sr"><span>Avg Speed</span><span class="sv g">${d.traffic.avg_speed} km/h</span></div>
      <div class="sr"><span>Speed Limit</span><span class="sv">${d.traffic.avg_speed_limit} km/h</span></div>
      <div class="sr"><span>Congestion %</span><span class="sv ${d.traffic.congestion_pct>60?'r':d.traffic.congestion_pct>30?'y':'g'}">${d.traffic.congestion_pct}%</span></div>
      <div class="sr"><span>Avg Occupancy</span><span class="sv y">${d.traffic.avg_occupancy}%</span></div>
    </div>

    <div class="rcard">
      <h4>⏰ Best Time to Depart</h4>
      ${depRows}
    </div>

    <div class="rcard">
      <h4>🕳️ Hazards on Route &nbsp;<span style="font-weight:normal;color:var(--gray)">(${d.hazards.total} total)</span></h4>
      <div class="sr" style="margin-bottom:6px">
        <span>🔴 High / 🟡 Medium / 🟢 Low</span>
        <span><span class="r">${d.hazards.high}</span> / <span class="y">${d.hazards.medium}</span> / <span class="g">${d.hazards.low}</span></span>
      </div>
      <div class="sr" style="margin-bottom:8px"><span>Road Quality Score</span>
        <span class="sv" style="color:${safetyColor(d.hazards.road_quality)}">${d.hazards.road_quality}/100</span>
      </div>
      ${hzHtml}
    </div>

    <div class="rcard"><h4>⚠️ Active Incidents &nbsp;<span style="font-weight:normal;color:var(--gray)">(${d.incidents.total})</span></h4>${incHtml}</div>

    <div class="rcard">
      <h4>📍 POIs Along Route &nbsp;<span style="font-weight:normal;color:var(--gray)">(${d.pois.total} found)</span></h4>
      <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px">
        ${Object.entries(d.pois.by_cat).map(([c,n])=>`<span class="tag t-${c}">${catEmoji[c]||'📍'} ${n} ${c.replace(/_/g,' ')}</span>`).join('')}
      </div>
      ${poiHtml}
    </div>

    <div class="rcard">
      <h4>🤖 ML Prediction</h4>
      <div class="sr"><span>Model</span><span class="sv t">${d.ml.model_name}</span></div>
      <div class="sr"><span>Prediction</span><span class="sv ${d.ml.congestion?'r':'g'}">${d.ml.congestion?'🔴 Congested':'🟢 Clear'}</span></div>
      <div class="sr"><span>Confidence</span><span class="sv b">${d.ml.confidence}%</span></div>
      <div class="sr"><span>Features</span><span class="sv" style="font-size:.7rem">14 features (speed, count, weather, hour…)</span></div>
    </div>`;
}

// ── Traffic Overview Panel ────────────────────────────────────────────────────
async function loadTrafficPanel(){
  if(document.getElementById('traffic-panel').dataset.loaded) return;
  const res=await fetch('/traffic_overview');
  const d=await res.json();
  const hours=Array.from({length:24},(_,i)=>i);
  const bars=hours.map(h=>{
    const c=d.hourly_congestion[h]||0; const s=d.hourly_speed[h]||50;
    const col=c>0.6?'#EF4444':c>0.3?'#F59E0B':'#10B981';
    return `<div style="display:flex;flex-direction:column;align-items:center;flex:1">
      <div style="height:${Math.round(c*60)}px;width:100%;background:${col};border-radius:2px 2px 0 0;min-height:2px;position:relative">
        <div style="position:absolute;bottom:100%;left:50%;transform:translateX(-50%);font-size:.52rem;color:${col};white-space:nowrap">${Math.round(c*100)}%</div>
      </div>
      <div style="font-size:.55rem;color:#64748B;margin-top:2px">${h}</div>
    </div>`;
  }).join('');
  document.getElementById('traffic-panel').innerHTML=`
    <div style="padding:10px">
      <div class="rcard">
        <h4>📊 Live Traffic Stats</h4>
        <div class="sr"><span>Total Roads</span><span class="sv b">${d.total_roads}</span></div>
        <div class="sr"><span>Total Records</span><span class="sv b">${d.total_records.toLocaleString()}</span></div>
        <div class="sr"><span>Currently Congested</span><span class="sv r">${d.currently_congested}</span></div>
        <div class="sr"><span>Avg Network Speed</span><span class="sv g">${d.avg_speed} km/h</span></div>
        <div class="sr"><span>Peak Congestion Hour</span><span class="sv y">${d.peak_hour}:00</span></div>
        <div class="sr"><span>Best Travel Hour</span><span class="sv g">${d.best_hour}:00</span></div>
      </div>
      <div class="rcard">
        <h4>⏰ Hourly Congestion Rate</h4>
        <div style="display:flex;align-items:flex-end;height:80px;gap:1px;padding:4px 0">${bars}</div>
        <div style="font-size:.65rem;color:#64748B;margin-top:4px;text-align:center">Hour of Day (0–23)</div>
      </div>
      <div class="rcard">
        <h4>🛣️ Road Type Breakdown</h4>
        ${d.by_type.map(r=>`<div class="sr"><span>${r.type}</span><span class="sv">${r.count} roads · ${r.avg_cong}% cong</span></div>`).join('')}
      </div>
      <div class="rcard">
        <h4>🌦️ Weather Impact on Speed</h4>
        ${d.weather_impact.map(w=>`<div class="sr"><span>${w.condition}</span><span class="sv">${w.avg_speed} km/h avg</span></div>`).join('')}
      </div>
    </div>`;
  document.getElementById('traffic-panel').dataset.loaded='1';
}

// ── History log ───────────────────────────────────────────────────────────────
function saveToHistory(d){
  const r=d.routes[selectedMode];
  const entry={
    time: new Date().toLocaleTimeString(),
    src: srcSel.options[srcSel.selectedIndex]?.text || `${srcLL.lat.toFixed(3)},${srcLL.lon.toFixed(3)}`,
    dst: dstSel.options[dstSel.selectedIndex]?.text || `${dstLL.lat.toFixed(3)},${dstLL.lon.toFixed(3)}`,
    mode: selectedMode, eta: r.eta_min, dist: r.distance_km, safety: r.safety_score
  };
  journeyLog.unshift(entry);
  const panel=document.getElementById('history-panel');
  panel.innerHTML=journeyLog.slice(0,10).map(e=>`
    <div class="log-item">
      <div style="font-weight:700;font-size:.78rem">${e.src} → ${e.dst}</div>
      <div style="font-size:.68rem;color:#64748B;margin-top:2px">${e.time} · ${e.mode} · ${e.eta} min · ${e.dist} km · Safety: ${e.safety}</div>
    </div>`).join('');
}
</script>
</body>
</html>
"""

# ── API ────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(
        HTML,
        landmarks=LANDMARKS,
        hourly_cong={str(k): round(float(v),3) for k,v in hourly_cong.items()},
        hourly_speed={str(k): round(float(v),1) for k,v in hourly_speed.items()},
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    body     = request.get_json()
    src_lat  = float(body["src_lat"]); src_lon = float(body["src_lon"])
    dst_lat  = float(body["dst_lat"]); dst_lon = float(body["dst_lon"])

    corr     = get_corridor(src_lat, src_lon, dst_lat, dst_lon)
    weather  = get_weather()

    # hazards
    ph = pothole_df.copy()
    ph["_sd"] = seg_dist_vec(ph.latitude.values, ph.longitude.values, src_lon, src_lat, dst_lon, dst_lat)
    rh = ph[ph["_sd"] < 1500].sort_values("_sd").head(30).copy()
    hc = rh["severity"].value_counts().to_dict()

    # POIs
    pf = poi_df.copy()
    pf["_sd"] = seg_dist_vec(pf.latitude.values, pf.longitude.values, src_lon, src_lat, dst_lon, dst_lat)
    rp = pf[pf["_sd"] < 1000].sort_values("_sd").head(40).copy()
    rp["dist_m"] = rp["_sd"]
    by_cat = rp["category"].value_counts().to_dict()

    # incidents
    inc = incident_df.copy()
    inc["_sd"] = seg_dist_vec(inc.latitude.values, inc.longitude.values, src_lon, src_lat, dst_lon, dst_lat)
    ri = inc[(inc["_sd"] < 1500) & (~inc["is_cleared"].astype(bool))].sort_values("_sd").head(10).copy()

    routes = compute_routes(src_lat, src_lon, dst_lat, dst_lon, corr, weather, rh, ri)

    # traffic stats
    avg_speed  = float(corr.average_speed_kmph.mean())  if len(corr) else 45.0
    avg_lim    = float(corr.speed_limit_kmph.mean())    if len(corr) else 50.0
    avg_occ    = float(corr.occupancy.mean())            if "occupancy" in corr.columns and len(corr) else 0.5
    cong_pct   = int(corr.congestion.mean()*100)         if len(corr) else 30

    # heatmap points (sample from corridor)
    hmap = []
    for _, row in corr.iterrows():
        hmap.append({"lat": float(row.latitude), "lon": float(row.longitude),
                     "level": str(row.get("congestion_level","free"))})

    # ML
    hour = datetime.now().hour; dow = datetime.now().weekday()
    is_peak = 1 if hour in list(range(7,11))+list(range(17,21)) else 0
    is_wknd = 1 if dow >= 5 else 0
    sp_ratio = round(avg_speed/max(avg_lim,1), 3)
    flow     = round(min(1.0, float(corr.vehicle_count.mean() if len(corr) else 100)/400), 3)
    road_enc = 1
    w_enc    = {"clear":0,"drizzle":1,"rain":2,"heavy_rain":3,"fog":4}.get(weather["condition"],0)
    X = [[float(corr.vehicle_count.mean() if len(corr) else 100), avg_speed, avg_lim, 4,
          avg_occ, hour, dow, is_peak, is_wknd, sp_ratio, flow, road_enc, w_enc, datetime.now().month]]
    try:    proba = ml_model.predict_proba(scaler.transform(X))[0]
    except: proba = ml_model.predict_proba(X)[0]

    result = {
        "routes":   routes,
        "weather":  weather,
        "heatmap":  hmap,
        "traffic": {
            "roads_monitored": len(corr),
            "congested_roads": int(corr.congestion.sum()) if len(corr) else 0,
            "avg_speed":       round(avg_speed, 1),
            "avg_speed_limit": round(avg_lim, 1),
            "congestion_pct":  cong_pct,
            "avg_occupancy":   round(avg_occ*100, 1),
        },
        "hazards": {
            "total": len(rh), "high": int(hc.get("high",0)),
            "medium": int(hc.get("medium",0)), "low": int(hc.get("low",0)),
            "road_quality": road_quality_score(rh),
            "list": rh[["hazard_id","hazard_type","severity","latitude","longitude",
                        "confidence_score","depth_cm","repair_cost_inr"]].rename(
                    columns={"repair_cost_inr":"repair_cost"}).fillna(0).to_dict("records"),
        },
        "incidents": {
            "total": len(ri),
            "list":  ri[["incident_id","incident_type","severity","latitude","longitude",
                         "delay_added_min","road_name"]].to_dict("records"),
        },
        "pois": {
            "total":  len(rp),
            "by_cat": {k:int(v) for k,v in by_cat.items()},
            "list":   rp[["poi_id","name","category","latitude","longitude",
                          "rating","dist_m","open_now","open_hours"]].to_dict("records"),
        },
        "best_departure": best_departure_times(),
        "ml": {
            "congestion":  int(proba[1] > 0.5),
            "confidence":  int(max(proba)*100),
            "model_name":  model_name,
        },
    }
    return safe_json(result)


@app.route("/traffic_overview")
def traffic_overview():
    peak_hour = int(max(hourly_cong, key=hourly_cong.get))
    best_hour = int(min(hourly_cong, key=hourly_cong.get))
    by_type   = []
    for rtype, grp in latest.groupby("road_type"):
        by_type.append({
            "type":     str(rtype),
            "count":    int(len(grp)),
            "avg_cong": int(grp.congestion.mean()*100),
        })
    wi = []
    for wc, grp in traffic_df.groupby("weather"):
        wi.append({"condition":str(wc), "avg_speed":round(float(grp.average_speed_kmph.mean()),1)})
    result = {
        "total_roads":         int(latest.road_id.nunique()),
        "total_records":       int(len(traffic_df)),
        "currently_congested": int(latest.congestion.sum()),
        "avg_speed":           round(float(latest.average_speed_kmph.mean()),1),
        "peak_hour":           peak_hour,
        "best_hour":           best_hour,
        "hourly_congestion":   {str(k):round(float(v),3) for k,v in hourly_cong.items()},
        "hourly_speed":        {str(k):round(float(v),1) for k,v in hourly_speed.items()},
        "by_type":             by_type,
        "weather_impact":      wi,
    }
    return safe_json(result)


if __name__ == "__main__":
    print("\n" + "="*52)
    print("  TrafficADS v3.0 — Smart Route Planner")
    print("  Open: http://localhost:5000")
    print("="*52 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)

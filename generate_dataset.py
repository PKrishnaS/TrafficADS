"""
generate_dataset.py  v2.0 — UPGRADED
Production-scale synthetic datasets for Chennai:
  traffic_data.csv   — 50,000 sensor readings (real road names, weather, lanes)
  pothole_data.csv   — 1,200 hazards
  poi_data.csv       — 800 POIs across 12 categories
  road_network.csv   — 80 road segments with speed limits & geometry
  weather_data.csv   — 8,760 hourly weather records (full year)
  incidents_data.csv — 400 traffic incidents
"""
import numpy as np, pandas as pd, os, random
from datetime import datetime, timedelta

random.seed(42); np.random.seed(42)
os.makedirs("data", exist_ok=True)

CHENNAI_ROADS = [
    ("RD_001","Anna Salai (Mount Road)",13.0690,80.2553,60,6,"arterial"),
    ("RD_002","GST Road",12.9700,80.1900,80,6,"highway"),
    ("RD_003","OMR (Old Mahabalipuram Road)",12.9200,80.2300,60,4,"arterial"),
    ("RD_004","ECR (East Coast Road)",12.8500,80.2500,80,4,"highway"),
    ("RD_005","NH-16 (Chennai-Kolkata Highway)",13.1500,80.2800,100,6,"highway"),
    ("RD_006","Rajiv Gandhi Salai (IT Corridor)",12.9600,80.2400,60,4,"arterial"),
    ("RD_007","Poonamallee High Road",13.0500,80.1500,60,4,"arterial"),
    ("RD_008","Grand Southern Trunk Road",12.9800,80.1100,60,4,"arterial"),
    ("RD_009","Inner Ring Road",13.0200,80.2100,60,6,"ring"),
    ("RD_010","Outer Ring Road",13.0000,80.1500,80,6,"ring"),
    ("RD_011","Velachery Main Road",12.9800,80.2200,50,4,"arterial"),
    ("RD_012","Sardar Patel Road",13.0067,80.2206,60,4,"arterial"),
    ("RD_013","Cathedral Road",13.0600,80.2500,50,4,"local"),
    ("RD_014","Nungambakkam High Road",13.0580,80.2420,50,4,"local"),
    ("RD_015","Arcot Road",13.0500,80.2100,50,4,"local"),
    ("RD_016","Patel Road T Nagar",13.0400,80.2340,40,4,"local"),
    ("RD_017","Usman Road T Nagar",13.0418,80.2380,40,4,"local"),
    ("RD_018","LB Road Adyar",13.0000,80.2560,50,4,"local"),
    ("RD_019","100 Feet Road Vadapalani",13.0520,80.2050,60,4,"arterial"),
    ("RD_020","Kamarajar Salai Marina",13.0600,80.2830,60,6,"arterial"),
]
for i in range(21,81):
    lat=12.85+random.uniform(0,0.35); lon=80.10+random.uniform(0,0.30)
    spd=random.choice([40,50,60,80]); lanes=random.choice([2,4,6])
    rtype=random.choice(["local","arterial","highway"])
    CHENNAI_ROADS.append((f"RD_{i:03d}",f"Road Segment {i}",lat,lon,spd,lanes,rtype))

pd.DataFrame(CHENNAI_ROADS,columns=["road_id","road_name","lat","lon","speed_limit","lanes","road_type"]).to_csv("data/road_network.csv",index=False)
print(f"✓ road_network.csv  (80 roads)")

# Traffic — 50K rows
print("Generating traffic_data.csv (50,000 rows)...")
base=datetime(2024,1,1); records=[]
for _ in range(50000):
    road=random.choice(CHENNAI_ROADS)
    rid,rname,rlat,rlon,slimit,lanes,rtype=road
    ts=base+timedelta(minutes=random.randint(0,525600))
    h=ts.hour; dow=ts.weekday()
    is_peak=(7<=h<=10)or(17<=h<=20); is_night=h<5 or h>22; is_wknd=dow>=5
    dens=random.uniform(0.1,0.9)
    if is_peak and not is_wknd: dens=min(1.0,dens+random.uniform(0.3,0.6))
    if is_night: dens=max(0.0,dens-random.uniform(0.3,0.5))
    if is_wknd: dens=max(0.0,dens-random.uniform(0.1,0.3))
    mvc=lanes*80; vc=int(max(0,min(mvc,mvc*dens+random.gauss(0,10))))
    spd=max(3,min(slimit+10,slimit*(1-dens*0.75)+random.gauss(0,3)))
    weather=random.choices(["clear","rain","fog","drizzle"],[60,20,10,10])[0]
    wf={"clear":1.0,"rain":0.75,"fog":0.65,"drizzle":0.88}[weather]
    spd*=wf
    cong=1 if spd<slimit*0.4 or vc>mvc*0.75 else 0
    clvl="heavy" if spd<slimit*0.3 else("moderate" if spd<slimit*0.6 else "free")
    records.append({
        "timestamp":ts.strftime("%Y-%m-%d %H:%M:%S"),
        "road_id":rid,"road_name":rname,"road_type":rtype,
        "latitude":round(rlat+random.gauss(0,0.003),6),
        "longitude":round(rlon+random.gauss(0,0.003),6),
        "vehicle_count":vc,"average_speed_kmph":round(spd,1),
        "speed_limit_kmph":slimit,"lanes":lanes,
        "occupancy":round(min(1.0,dens+random.gauss(0,0.05)),3),
        "congestion":cong,"congestion_level":clvl,"weather":weather,
        "hour":h,"day_of_week":dow,"is_peak_hour":int(is_peak),"is_weekend":int(is_wknd),
    })
tdf=pd.DataFrame(records); tdf.to_csv("data/traffic_data.csv",index=False)
print(f"✓ traffic_data.csv  (50,000 rows)  Congestion: {tdf['congestion'].mean():.1%}")

# Hazards — 1200
print("Generating pothole_data.csv (1,200 rows)...")
HTYPES={"pothole":0.40,"road_crack":0.20,"speed_breaker":0.15,"debris":0.10,"waterlogging":0.08,"road_collapse":0.04,"oil_spill":0.03}
SEVS=["low","medium","high"]; SCOSTS={"pothole":[100,200,400],"road_crack":[50,150,250],"speed_breaker":[30,100,200],"debris":[20,80,150],"waterlogging":[80,200,350],"road_collapse":[200,500,800],"oil_spill":[50,150,300]}
hrows=[]
for i in range(1200):
    road=random.choice(CHENNAI_ROADS)
    ht=random.choices(list(HTYPES),weights=list(HTYPES.values()))[0]
    sv=random.choices(SEVS,[0.4,0.4,0.2])[0]; si=SEVS.index(sv)
    hrows.append({"hazard_id":f"HZ_{i:04d}","road_id":road[0],"road_name":road[1],
        "latitude":round(road[2]+random.gauss(0,0.005),6),"longitude":round(road[3]+random.gauss(0,0.005),6),
        "hazard_type":ht,"severity":sv,"depth_cm":round(random.uniform(1,25),1) if ht=="pothole" else None,
        "area_sqm":round(random.uniform(0.1,4.0),2),"confidence_score":round(random.uniform(0.60,0.99),2),
        "reported_at":(base+timedelta(days=random.randint(0,365),hours=random.randint(0,23))).strftime("%Y-%m-%d %H:%M"),
        "repair_cost_inr":SCOSTS[ht][si]+random.randint(-20,50),"is_repaired":random.choices([0,1],[0.7,0.3])[0]})
pd.DataFrame(hrows).to_csv("data/pothole_data.csv",index=False)
print(f"✓ pothole_data.csv  (1,200 rows)")

# POI — 800
print("Generating poi_data.csv (800 locations)...")
POI_CAT={
    "hospital":(["Apollo Hospital","Fortis Malar","MIOT International","Govt General Hospital","Stanley Medical","Sri Ramachandra","SRM Hospital","Vijaya Hospital"],70),
    "fuel_station":(["Indian Oil","BPCL","HP Petrol Bunk","Shell","Reliance","Essar","Bharat Petroleum"],80),
    "shop":(["DMart","Big Bazaar","Nilgiris","Spencer's","Saravana Stores","More Supermarket","Reliance Fresh","Lifestyle","Croma","Poorvika"],120),
    "restaurant":(["Saravana Bhavan","Murugan Idli","Anjappar","Buhari","KFC","McDonald's","Dominos","Burger King","Subway","Pizza Hut","Adyar Ananda Bhavan"],120),
    "atm":(["SBI ATM","HDFC ATM","ICICI ATM","Axis ATM","Canara ATM","PNB ATM"],80),
    "pharmacy":(["Apollo Pharmacy","MedPlus","Frank Ross","Netmeds","Guardian"],70),
    "school":(["DAV School","PSBB School","KV School","Don Bosco","Santhome","Vidya Mandir"],50),
    "parking":(["City Parking Zone","Mall Parking","Metro Parking","Multi-level Parking"],50),
    "bank":(["SBI Branch","HDFC Branch","ICICI Branch","Axis Bank","Canara Bank","Indian Bank"],50),
    "metro_station":(["Central Metro","Egmore Metro","Guindy Metro","Airport Metro","Alandur Metro","Koyambedu Metro","Anna Nagar Metro"],40),
    "ev_station":(["Tata Power EV","Ather Grid","BPCL EV","ChargeZone","EESL Charging"],30),
    "police_station":(["Chennai Police Station","Traffic Police Post","Highway Patrol"],40),
}
prows=[]; pid=0
for cat,(names,count) in POI_CAT.items():
    for _ in range(count):
        road=random.choice(CHENNAI_ROADS)
        prows.append({"poi_id":f"POI_{pid:04d}","name":random.choice(names),"category":cat,
            "latitude":round(road[2]+random.gauss(0,0.01),6),"longitude":round(road[3]+random.gauss(0,0.01),6),
            "rating":round(random.uniform(2.8,5.0),1),"open_now":random.choices([True,False],[0.75,0.25])[0],
            "open_hours":random.choice(["24/7","6AM-10PM","8AM-9PM","9AM-8PM"]),"road_name":road[1]})
        pid+=1
pd.DataFrame(prows).to_csv("data/poi_data.csv",index=False)
print(f"✓ poi_data.csv      ({pid} rows, 12 categories)")

# Weather — 8760 hourly
print("Generating weather_data.csv (8,760 hourly rows)...")
wrows=[]
for h in range(8760):
    ts=base+timedelta(hours=h); mo=ts.month
    rp=0.35 if mo in [6,7,8,9,10,11] else 0.08
    wc=random.choices(["clear","drizzle","rain","heavy_rain","fog"],[0.55,0.15,rp,rp/3,0.05])[0]
    sf={"clear":1.0,"drizzle":0.90,"rain":0.75,"heavy_rain":0.60,"fog":0.65}.get(wc,1.0)
    wrows.append({"timestamp":ts.strftime("%Y-%m-%d %H:00:00"),"condition":wc,
        "temperature_c":round(22+10*np.sin(2*np.pi*(ts.hour-6)/24)+random.gauss(0,1.5),1),
        "humidity_pct":random.randint(50,95),
        "visibility_km":{"clear":10,"drizzle":7,"rain":4,"heavy_rain":2,"fog":1}.get(wc,10),
        "speed_factor":sf})
pd.DataFrame(wrows).to_csv("data/weather_data.csv",index=False)
print(f"✓ weather_data.csv  (8,760 rows)")

# Incidents — 400
print("Generating incidents_data.csv (400 rows)...")
irows=[]
for i in range(400):
    road=random.choice(CHENNAI_ROADS)
    ts=base+timedelta(hours=random.randint(0,8760)); dur=random.randint(15,240)
    it=random.choice(["accident","road_block","protest","vip_movement","construction","flood","breakdown"])
    sv=random.choice(["minor","major","critical"]); dm={"minor":5,"major":20,"critical":45}[sv]+random.randint(-3,10)
    irows.append({"incident_id":f"INC_{i:04d}","road_id":road[0],"road_name":road[1],
        "latitude":round(road[2]+random.gauss(0,0.003),6),"longitude":round(road[3]+random.gauss(0,0.003),6),
        "incident_type":it,"severity":sv,"start_time":ts.strftime("%Y-%m-%d %H:%M"),
        "duration_min":dur,"delay_added_min":max(0,dm),"lanes_affected":random.randint(1,road[5]),
        "is_cleared":random.choices([True,False],[0.65,0.35])[0]})
pd.DataFrame(irows).to_csv("data/incidents_data.csv",index=False)
print(f"✓ incidents_data.csv (400 rows)\n\n✅ All 6 datasets ready!")

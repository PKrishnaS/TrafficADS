"""
traffic_model.py  v2.0
Trains congestion prediction model on 50K rows with richer features.
"""
import pandas as pd, numpy as np, pickle, os, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

os.makedirs("models",exist_ok=True); os.makedirs("outputs",exist_ok=True)
print("="*55+"\n  TRAFFIC CONGESTION MODEL v2.0 (50K rows)\n"+"="*55)

df = pd.read_csv("data/traffic_data.csv", parse_dates=["timestamp"])
print(f"\n[1] Loaded {len(df):,} rows  |  Congestion rate: {df['congestion'].mean():.1%}")

df["hour"]       = df["timestamp"].dt.hour
df["month"]      = df["timestamp"].dt.month
df["speed_ratio"]= (df["average_speed_kmph"] / df["speed_limit_kmph"]).round(3)
df["flow_rate"]  = (df["vehicle_count"] / (df["lanes"] * 80)).round(3)
df["road_enc"]   = LabelEncoder().fit_transform(df["road_type"])
df["weather_enc"]= LabelEncoder().fit_transform(df["weather"])

feats = ["vehicle_count","average_speed_kmph","speed_limit_kmph","lanes","occupancy",
         "hour","day_of_week","is_peak_hour","is_weekend","speed_ratio","flow_rate",
         "road_enc","weather_enc","month"]
X = df[feats]; y = df["congestion"]

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
sc = StandardScaler(); X_tr_s=sc.fit_transform(X_tr); X_te_s=sc.transform(X_te)

models = {
    "Random Forest":  RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42),
    "Gradient Boost": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Reg":   LogisticRegression(max_iter=500, random_state=42),
}
results={}
for name, mdl in models.items():
    t0=time.time()
    Xf = X_tr_s if name=="Logistic Reg" else X_tr
    mdl.fit(Xf, y_tr)
    Xft = X_te_s if name=="Logistic Reg" else X_te
    pred=mdl.predict(Xft); prob=mdl.predict_proba(Xft)[:,1]
    acc=accuracy_score(y_te,pred); auc=roc_auc_score(y_te,prob)
    results[name]={"acc":acc,"auc":auc,"time":round(time.time()-t0,3),"model":mdl}
    print(f"\n  {name}: Acc={acc:.4f}  AUC={auc:.4f}  ({results[name]['time']}s)")

best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]["model"]
print(f"\n[Best] {best_name}")
with open("models/congestion_model.pkl","wb") as f:
    pickle.dump({"model":best,"scaler":sc,"features":feats,"model_name":best_name},f)
print("  Saved → models/congestion_model.pkl")

# Plot
fig,axes=plt.subplots(1,3,figsize=(15,5))
fig.suptitle("Traffic Congestion Prediction v2.0 — ADS Framework",fontsize=13,fontweight="bold")
names=list(results.keys()); accs=[results[n]["acc"] for n in names]; aucs=[results[n]["auc"] for n in names]
x=np.arange(len(names)); w=0.35
axes[0].bar(x-w/2,accs,w,label="Accuracy",color="#0D9488"); axes[0].bar(x+w/2,aucs,w,label="ROC-AUC",color="#F59E0B")
axes[0].set_xticks(x); axes[0].set_xticklabels(names,fontsize=8); axes[0].set_ylim(0,1.1)
axes[0].legend(); axes[0].set_title("Model Comparison")
for i,(a,u) in enumerate(zip(accs,aucs)):
    axes[0].text(i-w/2,a+0.01,f"{a:.3f}",ha="center",fontsize=7)
    axes[0].text(i+w/2,u+0.01,f"{u:.3f}",ha="center",fontsize=7)

if hasattr(best,"feature_importances_"):
    imp=best.feature_importances_; idx=np.argsort(imp)[::-1][:10]
    axes[1].barh(range(10),[imp[i] for i in idx],color="#3B82F6")
    axes[1].set_yticks(range(10)); axes[1].set_yticklabels([feats[i] for i in idx],fontsize=8)
    axes[1].set_title("Feature Importances"); axes[1].invert_yaxis()

hours=range(24); hourly=df.groupby("hour")["congestion"].mean()
axes[2].bar(hours,[hourly.get(h,0) for h in hours],color=["#EF4444" if (7<=h<=10 or 17<=h<=20) else "#10B981" for h in hours])
axes[2].set_title("Congestion Rate by Hour"); axes[2].set_xlabel("Hour"); axes[2].set_ylabel("Congestion Rate")
axes[2].axhline(0.5,color="orange",linestyle="--",linewidth=1)

plt.tight_layout(); plt.savefig("outputs/traffic_model_results.png",dpi=150,bbox_inches="tight")
print("  Saved → outputs/traffic_model_results.png\n✅ Done!")

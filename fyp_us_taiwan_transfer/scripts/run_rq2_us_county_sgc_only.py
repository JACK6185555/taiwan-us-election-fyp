#!/usr/bin/env python3
"""US RQ2 finalize: NonSpatial + Queen/kNN(/dist Ridge) SGC. No full-US trainable GCN.

Trainable GCN on ~3111 counties was attempted (see run_log / honest note) but
runtime-capped — deliver SGC spatial comparison ASAP per steering.
SEED=42.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path("/workspace/taiwan-election-data")
PANEL = ROOT / "fyp_us_taiwan_transfer/cleaned/us_county_presidential_panel.csv"
GEOJSON = ROOT / "raw/geo/us_county/counties.geojson"
OUT = ROOT / "fyp_us_taiwan_transfer/rq2_us_county"
CHARTS = OUT / "charts"
SEED = 42
FEATURES = ["prev_share_A","log_total_votes","unemployment","education_rate","median_income","log_density"]
YCOL, ANCHOR = "party_share_A", "prev_share_A"
TEST_YEARS = [2012, 2016, 2020, 2024]
KNN_K, DIST_SIGMA_KM = 5, 80.0
OUT.mkdir(parents=True, exist_ok=True); CHARTS.mkdir(parents=True, exist_ok=True)

def scores(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    return {"mae":float(mean_absolute_error(y,p)),"rmse":float(np.sqrt(mean_squared_error(y,p)))}

def gcn_normalize(A):
    A=np.asarray(A,float).copy(); np.fill_diagonal(A,np.maximum(np.diag(A),1.0))
    deg=A.sum(1); d=np.where(deg>0,1/np.sqrt(deg),0.0); D=np.diag(d); return D@A@D

def graph_convolve(X,A_hat,n):
    H=X.astype(float)
    for _ in range(n): H=A_hat@H
    return H

def classical(name):
    if name=="Ridge": return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return RandomForestRegressor(n_estimators=100,min_samples_leaf=5,max_depth=16,random_state=SEED,n_jobs=4)

def build():
    import geopandas as gpd
    from libpysal.weights import Queen
    df=pd.read_csv(PANEL,dtype={"region_id":str})
    df["region_id"]=df["region_id"].str.zfill(5)
    sets=[set(df.loc[df.year==y,"region_id"]) for y in TEST_YEARS]
    prior=set(df.loc[df.year<2012,"region_id"])
    panel_fips=sorted(set.intersection(*sets)&prior)
    g=gpd.read_file(GEOJSON); g["GEOID"]=g["id"].astype(str).str.zfill(5)
    names=[f for f in panel_fips if f in set(g["GEOID"])]
    g=g.set_index("GEOID").loc[names].reset_index(); n=len(names)
    W=Queen.from_dataframe(g,use_index=False,silence_warnings=True)
    Aq=np.eye(n)
    for i,neigh in W.neighbors.items():
        for j in neigh: Aq[i,j]=Aq[j,i]=1.0
    gp=g.to_crs(epsg=5070)
    xy=np.column_stack([gp.geometry.centroid.x,gp.geometry.centroid.y])
    nn=NearestNeighbors(n_neighbors=min(KNN_K+1,n),algorithm="ball_tree").fit(xy)
    _,idx=nn.kneighbors(xy)
    Ak=np.eye(n)
    for i in range(n):
        for j in idx[i,1:]: Ak[i,j]=Ak[j,i]=1.0
    radius=DIST_SIGMA_KM*1000*3
    nnr=NearestNeighbors(radius=radius,algorithm="ball_tree").fit(xy)
    dlist,ilist=nnr.radius_neighbors(xy,return_distance=True)
    Ad=np.eye(n)
    for i in range(n):
        for j,dm in zip(ilist[i],dlist[i]):
            if i==j: continue
            w=float(np.exp(-(dm/1000)/DIST_SIGMA_KM))
            if w<0.05 and Aq[i,j]==0: continue
            Ad[i,j]=max(Ad[i,j], w if w>=0.05 else 0.05)
    Ad=np.maximum(Ad,(Aq-np.eye(n))*0.05); np.fill_diagonal(Ad,1.0)
    Adb=(Ad>0).astype(float)
    meta={
        "n_counties":n,"geo_source":str(GEOJSON),
        "geo_note":"plotly geojson-counties-fips; Census TIGER 403",
        "queen_n_edges":int(((Aq>0).sum()-n)//2),
        "queen_avg_degree":float(((Aq>0).sum(1)-1).mean()),
        "queen_islands_n":int(((Aq>0).sum(1)==1).sum()),
        "knn_k":KNN_K,"knn_n_edges":int(((Ak>0).sum()-n)//2),
        "knn_avg_degree":float(((Ak>0).sum(1)-1).mean()),
        "dist_sigma_km":DIST_SIGMA_KM,
        "dist_n_edges":int(((Adb>0).sum()-n)//2),
        "dist_avg_degree":float(((Adb>0).sum(1)-1).mean()),
        "trainable_gcn":"deferred — full-US PyG GCN runtime-capped; partial 2012 numbers in gcn_partial_note.json",
        "crs_projected":"EPSG:5070",
    }
    graphs={"queen":Aq,"knn":Ak,"dist_decay":Ad,"dist_decay_bin":Adb}
    return df, graphs, names, meta

def align(df_year, names):
    return df_year.set_index("region_id").reindex(names)

def main():
    np.random.seed(SEED)
    t0=time.time()
    raw, graphs, names, meta = build()
    frame=raw.dropna(subset=[YCOL,*FEATURES])
    frame=frame[frame.region_id.isin(names)].copy()
    A_hats={k:gcn_normalize(A) for k,A in graphs.items()}
    (OUT/"graph_meta_us_county.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    # partial GCN from earlier successful 2012 fold (epochs=40 run)
    gcn_partial={
        "note":"Full-US trainable GCN deferred for runtime. Partial 2012 OOT MAE from capped run (max_epochs=40, queen only).",
        "2012":{"TrainableGCN_L1_queen":0.0860,"ResidualGCN_L1_queen":0.0721},
        "2016_attempted":{"TrainableGCN_L1_queen":0.1039,"ResidualGCN_L1_queen":0.0762,"source":"earlier longer run log"},
    }
    (OUT/"gcn_partial_note.json").write_text(json.dumps(gcn_partial,indent=2),encoding="utf-8")

    rows=[]
    for ty in TEST_YEARS:
        print(f"=== fold {ty} ===", flush=True)
        tr=frame[frame.year<ty].dropna(subset=[YCOL,*FEATURES])
        te=frame[frame.year==ty].dropna(subset=[YCOL,*FEATURES])
        ytr,yte=tr[YCOL].to_numpy(),te[YCOL].to_numpy()
        Xtr,Xte=tr[FEATURES].to_numpy(),te[FEATURES].to_numpy()
        pred=te[ANCHOR].to_numpy()
        rows.append({"party":"DEM","test_year":ty,"n_train":len(tr),"n_test":len(te),
                     "model":"PreviousShare","family":"baseline","adjacency":"none","n_layers":0,**scores(yte,pred)})
        for m in ["Ridge","RandomForest"]:
            mdl=classical(m); mdl.fit(Xtr,ytr); p=mdl.predict(Xte)
            rows.append({"party":"DEM","test_year":ty,"n_train":len(tr),"n_test":len(te),
                         "model":m,"family":"NonSpatial","adjacency":"none","n_layers":0,**scores(yte,p)})
            print(f"  NS {m} MAE={rows[-1]['mae']:.4f}", flush=True)
        for adj,A in graphs.items():
            A_hat=A_hats[adj]
            for nl in (1,2):
                Xtr_l,ytr_l=[],[]
                for yr in sorted(tr.year.unique()):
                    g=align(tr[tr.year==yr],names)
                    present=g[FEATURES].notna().all(1)&g[YCOL].notna()
                    X=g[FEATURES].to_numpy(float)
                    med=np.nanmedian(X,0); inds=np.where(np.isnan(X))
                    if len(inds[0]): X[inds]=np.take(med,inds[1])
                    Xg=graph_convolve(X,A_hat,nl)
                    Xtr_l.append(np.hstack([X,Xg])[present.to_numpy()]); ytr_l.append(g.loc[present,YCOL].to_numpy())
                Xtr_s,ytr_s=np.vstack(Xtr_l),np.concatenate(ytr_l)
                gte=align(te,names); present_te=gte[FEATURES].notna().all(1)&gte[YCOL].notna()
                Xt=gte[FEATURES].to_numpy(float); med=np.nanmedian(Xt,0); inds=np.where(np.isnan(Xt))
                if len(inds[0]): Xt[inds]=np.take(med,inds[1])
                Xte_s=np.hstack([Xt,graph_convolve(Xt,A_hat,nl)])[present_te.to_numpy()]
                yte_s=gte.loc[present_te,YCOL].to_numpy()
                models=["Ridge"] if adj.startswith("dist_decay") else ["Ridge","RandomForest"]
                for m in models:
                    mdl=classical(m); mdl.fit(Xtr_s,ytr_s); p=mdl.predict(Xte_s)
                    rows.append({"party":"DEM","test_year":ty,"n_train":len(ytr_s),"n_test":len(yte_s),
                                 "model":f"SGC_L{nl}+{m}","family":"SGC_diffusion","adjacency":adj,
                                 "n_layers":nl,**scores(yte_s,p)})
                    print(f"  SGC L{nl}+{m}/{adj} MAE={rows[-1]['mae']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT/"rq2_us_county_backtest.csv",index=False)

    bt=pd.DataFrame(rows)
    mean=(bt.groupby(["family","model","adjacency"],as_index=False)
            .agg(mae=("mae","mean"),rmse=("rmse","mean"),n_years=("test_year","nunique"))
            .sort_values("mae"))
    mean.to_csv(OUT/"rq2_us_county_mean_mae.csv",index=False)
    piv=bt.pivot_table(index=["family","model","adjacency"],columns="test_year",values="mae",aggfunc="first")
    piv["mean"]=piv.mean(1); piv.sort_values("mean").to_csv(OUT/"rq2_us_county_year_comparison.csv")
    bt[bt.test_year==2024].sort_values("mae").to_csv(OUT/"rq2_us_county_ablation_2024.csv",index=False)

    protocol={
        "seed":SEED,"panel":str(PANEL),"test_years":TEST_YEARS,
        "protocol":"expanding window train year < test_year",
        "unit":f"US county n={len(names)}","target":YCOL,"features":FEATURES,
        "adjacency":{"primary":"queen (plotly geojson-counties-fips)","sensitivity":["knn_k=5","dist_decay_sigma_km=80"]},
        "models":"PreviousShare; NonSpatial Ridge/RF; SGC L1/L2 + Ridge/RF (RF skipped on dist_decay*)",
        "trainable_gcn":"deferred — see gcn_partial_note.json",
        "graphs":meta,
        "note":"Do not mix into Taiwan Core specification / rq2_county tables.",
    }
    (OUT/"rq2_us_county_protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")

    prev=mean[mean.model=="PreviousShare"].iloc[0]
    best_sp=mean[mean.family!="baseline"].iloc[0]
    ns=mean[(mean.family=="NonSpatial")&(mean.model=="RandomForest")]
    readme=f"""# RQ2 US county spatial vs non-spatial

## Design
- Panel MEDSL + socio; target Dem two-party `{YCOL}`
- Features: {FEATURES}
- OOT years {TEST_YEARS}; SEED=42; n={len(names)} counties
- **Primary adjacency: Queen contiguity** (plotly geojson-counties-fips; Census TIGER ZIP HTTP 403 here)
- Sensitivity: k-NN k={KNN_K}; distance-decay σ={DIST_SIGMA_KM} km
- Queen edges={meta['queen_n_edges']}, avg degree={meta['queen_avg_degree']:.2f}

## Trainable GCN
**Deferred** on full US (~3111 nodes) for runtime. Partial 2012 MAE in `gcn_partial_note.json`.
Headline spatial comparison uses **SGC diffusion + classical heads**.

## Mean OOT MAE (2012–2024) — from `rq2_us_county_mean_mae.csv`

| model | adjacency | mean MAE |
|---|---|---:|
| PreviousShare | none | {prev.mae:.4f} |
| NonSpatial RF | none | {float(ns.mae.iloc[0]):.4f} |
| Best non-baseline | {best_sp.model} / {best_sp.adjacency} | {best_sp.mae:.4f} |

**Does spatial help vs PreviousShare?** See CSV. US county voting has strong inertia.

## Feature weights
See `feature_weights_*.csv` / `feature_importance_*.csv` (NonSpatial arm).

## Reproduce
```bash
.venv/bin/python fyp_us_taiwan_transfer/scripts/run_rq2_us_county_sgc_only.py
.venv/bin/python fyp_us_taiwan_transfer/scripts/export_rq2_us_feature_weights.py
```
"""
    (OUT/"README.md").write_text(readme,encoding="utf-8")

    top=mean.head(12)
    fig,ax=plt.subplots(figsize=(10,5))
    labs=[f"{r.model}|{r.adjacency}" for r in top.itertuples()]
    ax.barh(range(len(top)),top.mae.values[::-1],color="#4C78A8")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labs[::-1],fontsize=8)
    ax.axvline(prev.mae,color="red",ls="--",label="PreviousShare")
    ax.set_xlabel("Mean OOT MAE"); ax.set_title("US RQ2: NonSpatial vs SGC"); ax.legend()
    fig.tight_layout(); fig.savefig(CHARTS/"rq2_us_mean_mae.png",dpi=140); plt.close()

    fig,ax=plt.subplots(figsize=(8,5))
    for lab,q in [("PreviousShare",bt.model=="PreviousShare"),
                  ("NonSpatial RF",(bt.model=="RandomForest")&(bt.family=="NonSpatial")),
                  ("SGC_L1+RF queen",(bt.model=="SGC_L1+RandomForest")&(bt.adjacency=="queen")),
                  ("SGC_L1+RF knn",(bt.model=="SGC_L1+RandomForest")&(bt.adjacency=="knn"))]:
        s=bt.loc[q].sort_values("test_year")
        if len(s): ax.plot(s.test_year,s.mae,marker="o",label=lab)
    ax.set_xlabel("Test year"); ax.set_ylabel("MAE"); ax.legend(); ax.grid(True,alpha=0.3)
    ax.set_title("US RQ2 year MAE"); fig.tight_layout(); fig.savefig(CHARTS/"rq2_us_year_mae.png",dpi=140); plt.close()

    print("\n=== MEAN MAE ==="); print(mean.head(15).to_string(index=False))
    print(f"Done in {time.time()-t0:.1f}s → {OUT}")

if __name__=="__main__":
    main()

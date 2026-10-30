# Education vs Economic vs Media — factor importance (Extension)
Built: 2026-10-30 16:31 Asia/Hong_Kong (UTC+8)
**Weights / ΔMAE = predictive dependence in OOT or national LOO — NOT causal.**
Family A locked headline (Prev MAE 0.0793) is **unchanged**; this folder is Extension analysis for teacher feedback.

## 1. Taiwan county (OOT 2012–2024)
| spec | model | mean MAE | Δ vs PreviousShare |
|---|---|---:|---:|
| Prev+Economic | Ridge | 0.0725 | -0.0067 |
| Prev+Education+Economic | Ridge | 0.0776 | -0.0017 |
| PreviousShare_only | baseline | 0.0793 | +0.0000 |
| Prev+Economic | RandomForest | 0.0882 | +0.0089 |
| Prev+Education | Ridge | 0.0890 | +0.0098 |
| Prev+Education | RandomForest | 0.0928 | +0.0135 |
| Prev+Education+Economic | RandomForest | 0.0946 | +0.0153 |
| Economic_only | Ridge | 0.1058 | +0.0265 |
| Economic_only | RandomForest | 0.1211 | +0.0419 |
| Education_only | Ridge | 0.1331 | +0.0538 |
| Education_only | RandomForest | 0.1436 | +0.0644 |

Group permutation (RF on Prev+Edu+Econ; MAE increase):

| factor_group | mean MAE increase |
|---|---:|
| previous | 0.0548 |
| economic | 0.0006 |
| education | -0.0002 |

Mean |standardized Ridge coef| by group (Prev+Edu+Econ):

| factor_group | mean |coef| |
|---|---:|
| previous | 0.1097 |
| education | 0.0116 |
| economic | 0.0029 |

## 2. US county (OOT ≥2012)
| spec | model | mean MAE | Δ vs PreviousShare |
|---|---|---:|---:|
| PreviousShare_only | baseline | 0.0345 | +0.0000 |
| Prev+Education | Ridge | 0.0424 | +0.0079 |
| Prev+Education+Economic | Ridge | 0.0438 | +0.0093 |
| Prev+Economic | Ridge | 0.0440 | +0.0095 |
| Prev+Education | RandomForest | 0.0451 | +0.0106 |
| Prev+Economic | RandomForest | 0.0463 | +0.0118 |
| Prev+Education+Economic | RandomForest | 0.0474 | +0.0129 |
| Education_only | RandomForest | 0.1179 | +0.0835 |
| Economic_only | RandomForest | 0.1197 | +0.0852 |
| Economic_only | Ridge | 0.1286 | +0.0941 |
| Education_only | Ridge | 0.1303 | +0.0958 |

Group permutation (RF on Prev+Edu+Econ; MAE increase):

| factor_group | mean MAE increase |
|---|---:|
| previous | 0.1284 |
| economic | 0.0014 |
| education | 0.0007 |

Mean |standardized Ridge coef| by group (Prev+Edu+Econ):

| factor_group | mean |coef| |
|---|---:|
| previous | 0.1369 |
| education | 0.0083 |
| economic | 0.0070 |

## 3. National media / fundamentals (Extension; LOO)
| country | spec | n_years | MAE_LOO |
|---|---|---:|---:|
| TW | EconNational_only | 7 | 0.0748 |
| TW | PreviousShare_only | 6 | 0.0824 |
| TW | Prev+EconNational | 6 | 0.0843 |
| TW | Prev+MediaCampGap | 2 | 0.1596 |
| TW | MediaCampGap_only | 2 | 0.1596 |
| TW | Prev+MediaIssueSent | 2 | 0.1596 |
| US | Prev+MediaCampGap | 2 | 0.0169 |
| US | MediaCampGap_only | 2 | 0.0169 |
| US | Prev+MediaIssueSent | 2 | 0.0169 |
| US | Prev+EconNational+CCI | 5 | 0.0193 |
| US | EconNational_only | 5 | 0.0207 |
| US | Prev+EconNational | 5 | 0.0223 |
| US | PreviousShare_only | 5 | 0.0285 |

Media uses official X search pilot samples (small n posts/years). Do **not** treat Pearson≈1 on n=2 as evidence.

## 4. Empirical ranking (honest)
1. **Previous / partisan baseline** dominates county OOT in both TW and US (group permutation and |coef|).
2. Between **education** and **economic** county features added to PreviousShare: compare ΔMAE and group permutation in tables above — typically both add little vs PreviousShare alone; relative rank may differ by country.
3. **Media** is national Extension only; cannot enter county Family A; n_years≤2 for camp gap → descriptive / pilot only (cf. Brito et al. social-media prediction caution).

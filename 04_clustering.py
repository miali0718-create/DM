"""
04 國家分群（K-Means）
目的：把國家依「接種表現」與「死亡負擔」分成幾種類型，再針對每一類提出衛教／政策策略。
特徵（每國一筆）：
  - final_vacc      最新至少一劑接種率
  - completion      完整接種 / 至少一劑（打完第二劑的比例，反映後續追蹤）
  - days_to_30      從 2020/12/19 起到接種率達 30% 的天數（推動速度；未達到者設為觀察期上限）
  - log_deaths_all  log(1 + 平均每日每百萬人死亡)
  - death_trend     最後 60 天 vs 之前 的死亡數比值（log），< 0 代表死亡下降
輸出：figures/fig07~fig08、results/cluster_*.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import PROCESSED, FIG, RESULTS, RANDOM_STATE, setup_font

setup_font()
d = pd.read_csv(PROCESSED / "daily_clean.csv", parse_dates=["date"])
START, END = d["date"].min(), d["date"].max()
cut = END - pd.Timedelta(days=60)

rows = []
for iso, g in d.groupby("iso_code"):
    g = g.sort_values("date")
    if g["date"].max() < pd.Timestamp("2021-08-01"):
        continue  # 資料太早就停止更新的國家，最新接種率不具代表性
    reached = g.loc[g["vacc_rate"] >= 30, "date"]
    recent = g.loc[g["date"] >= cut, "deaths_per_million"].mean()
    before = g.loc[g["date"] < cut, "deaths_per_million"].mean()
    rows.append({
        "iso_code": iso, "country": g["country"].iat[0], "continent": g["continent"].iat[0],
        "final_vacc": g["vacc_rate"].iat[-1],
        "completion": g["full_rate"].iat[-1] / max(g["vacc_rate"].iat[-1], 1e-6),
        "days_to_30": (reached.iat[0] - START).days if len(reached) else (END - START).days + 30,
        "log_deaths_all": np.log1p(g["deaths_per_million"].mean()),
        "death_trend": np.log((recent + 0.05) / (before + 0.05)) if pd.notna(before) else 0.0,
    })
c = pd.DataFrame(rows).dropna()
features = ["final_vacc", "completion", "days_to_30", "log_deaths_all", "death_trend"]
X = StandardScaler().fit_transform(c[features])

# 用 Elbow + Silhouette + 穩定性 選 k
# 穩定性：隨機抽 80% 國家重新分群 50 次，和全資料的分群結果比較（Adjusted Rand Index，1 = 完全一致）
rng = np.random.default_rng(RANDOM_STATE)
ks = range(2, 8)
inertia, sil, stab = [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    aris = []
    for _ in range(50):
        idx = rng.choice(len(X), int(len(X) * 0.8), replace=False)
        sub = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e6))).fit(X[idx])
        aris.append(adjusted_rand_score(km.labels_[idx], sub.labels_))
    stab.append(np.mean(aris))
pd.DataFrame({"k": ks, "inertia": inertia, "silhouette": sil, "stability_ARI": stab}) \
  .round(4).to_csv(RESULTS / "cluster_k_selection.csv", index=False)
K = 4  # 輪廓係數、穩定性與可解釋性的折衷；可依 results/cluster_k_selection.csv 調整
print("silhouette:", dict(zip(ks, np.round(sil, 3))))
print("stability (ARI):", dict(zip(ks, np.round(stab, 3))))

km = KMeans(n_clusters=K, n_init=50, random_state=RANDOM_STATE).fit(X)
c["cluster"] = km.labels_

# 依群組特徵命名（用排序決定，避免每次重跑編號不同）
prof = c.groupby("cluster")[features].mean()
names = {}
rest = list(prof.index)
hi_death = prof.loc[rest, "log_deaths_all"].idxmax(); names[hi_death] = "高死亡後改善型"; rest.remove(hi_death)
hi_vacc = prof.loc[rest, "final_vacc"].idxmax(); names[hi_vacc] = "高接種穩定型"; rest.remove(hi_vacc)
low_comp = prof.loc[rest, "completion"].idxmin(); names[low_comp] = "第二劑落後型"; rest.remove(low_comp)
for cl in rest:
    names[cl] = "接種緩慢型"
c["cluster_name"] = c["cluster"].map(names)

profile = (c.groupby("cluster_name")
           .agg(國家數=("iso_code", "size"),
                最新接種率=("final_vacc", "mean"),
                完整接種比例=("completion", "mean"),
                達30pct天數=("days_to_30", "mean"),
                平均每日每百萬人死亡=("log_deaths_all", lambda s: np.expm1(s).mean()),
                近60天死亡趨勢=("death_trend", "mean"))
           .round(2))
profile.to_csv(RESULTS / "cluster_profile.csv", encoding="utf-8-sig")
c.sort_values(["cluster_name", "final_vacc"], ascending=[True, False]).round(3) \
 .to_csv(RESULTS / "cluster_members.csv", index=False, encoding="utf-8-sig")
print(profile.to_string())
for n, g in c.groupby("cluster_name"):
    print(n, "：", ", ".join(g.sort_values("final_vacc", ascending=False)["country"].head(10)))

# fig07：k 選擇
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].plot(list(ks), inertia, marker="o"); ax[0].set_title("Elbow（組內平方和）"); ax[0].set_xlabel("k")
ax[1].plot(list(ks), sil, marker="o", color="#d1495b"); ax[1].set_title("Silhouette 輪廓係數"); ax[1].set_xlabel("k")
ax[2].plot(list(ks), stab, marker="o", color="#66a182"); ax[2].set_title("穩定性（抽 80% 國家重分群，ARI）"); ax[2].set_xlabel("k")
for a in ax:
    a.axvline(K, color="#999", ls="--")
fig.savefig(FIG / "fig07_cluster_k_selection.png")
plt.close(fig)

# fig08：PCA 2D + 群組雷達/長條
pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
P = pca.transform(X)
palette = ["#d1495b", "#00798c", "#edae49", "#30638e", "#66a182", "#8d6a9f", "#444"]
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
for i, (n, g) in enumerate(c.groupby("cluster_name")):
    idx = g.index.map(c.index.get_loc)
    ax[0].scatter(P[idx, 0], P[idx, 1], s=30, color=palette[i], label=f"{n}（{len(g)}）")
label_these = {"Singapore", "Japan", "Israel", "The United Kingdom", "Peru", "Nigeria", "India",
               "United States of America", "Brazil", "Republic of Korea", "Hungary", "Egypt"}
for i, row in c.reset_index(drop=True).iterrows():
    if row["country"] in label_these:
        ax[0].annotate(row["country"], P[i], fontsize=7, alpha=.8)
ax[0].set_xlabel(f"PC1（{pca.explained_variance_ratio_[0]:.0%}）")
ax[0].set_ylabel(f"PC2（{pca.explained_variance_ratio_[1]:.0%}）")
ax[0].set_title("國家分群結果（PCA 投影）")
ax[0].legend(frameon=False, fontsize=9)
z = pd.DataFrame(X, columns=features).assign(cluster_name=c["cluster_name"].values).groupby("cluster_name").mean()
z.columns = ["最新接種率", "完整接種比例", "達30%天數", "死亡負擔", "近期死亡趨勢"]
im = ax[1].imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax[1].set_xticks(range(z.shape[1]), z.columns, rotation=30)
ax[1].set_yticks(range(z.shape[0]), z.index)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax[1].text(j, i, f"{z.values[i, j]:.1f}", ha="center", va="center", fontsize=9)
ax[1].set_title("各群特徵（標準化平均，紅=高、藍=低）")
fig.colorbar(im, ax=ax[1], shrink=.7)
fig.savefig(FIG / "fig08_clusters.png")
plt.close(fig)

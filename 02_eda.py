"""
02 探索式資料分析（EDA）
- 全球接種與死亡趨勢
- 不同洲別比較（用 ISO 代碼分洲）
- 亞洲標竿學習對象（新加坡、日本等）
輸出：figures/fig01~fig03、results/continent_summary.csv、results/asia_benchmark.csv
"""
import matplotlib.pyplot as plt
import pandas as pd

from config import PROCESSED, FIG, RESULTS, CONTINENT_COLORS, setup_font

setup_font()
d = pd.read_csv(PROCESSED / "daily_clean.csv", parse_dates=["date"])
cm = pd.read_csv(PROCESSED / "country_month.csv")
months = sorted(cm["month"].unique())

# ---------- fig01 全球趨勢 ----------
glob = cm.groupby("month").agg(vacc=("vacc_rate", "median"), full=("full_rate", "median"),
                               deaths=("deaths_pm", "median"), n=("iso_code", "nunique"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(glob.index, glob["vacc"], marker="o", label="至少一劑")
ax[0].plot(glob.index, glob["full"], marker="o", label="完整接種")
ax[0].set_title("各國接種率中位數（%）")
ax[0].legend(frameon=False)
ax[1].bar(glob.index, glob["deaths"], color="#888")
ax[1].set_title("各國每日每百萬人死亡數中位數")
for a in ax:
    a.tick_params(axis="x", rotation=45)
fig.suptitle("全球趨勢（2020/12 ~ 2021/10）", fontsize=14, y=1.03)
fig.savefig(FIG / "fig01_global_trend.png")
plt.close(fig)

# ---------- fig02 洲別比較 ----------
cont = cm.groupby(["continent", "month"]).agg(vacc=("vacc_rate", "median"),
                                              deaths=("deaths_pm", "median")).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for c, g in cont.groupby("continent"):
    ax[0].plot(g["month"], g["vacc"], marker="o", ms=3, label=c, color=CONTINENT_COLORS[c])
    ax[1].plot(g["month"], g["deaths"], marker="o", ms=3, label=c, color=CONTINENT_COLORS[c])
ax[0].set_title("至少一劑接種率中位數（%）")
ax[1].set_title("每日每百萬人死亡數中位數")
for a in ax:
    a.tick_params(axis="x", rotation=45)
ax[1].legend(frameon=False, bbox_to_anchor=(1, 1))
fig.suptitle("不同洲別的接種率與死亡率", fontsize=14, y=1.03)
fig.savefig(FIG / "fig02_continent.png")
plt.close(fig)

latest = d.sort_values("date").groupby("iso_code").tail(1)
cont_sum = (latest.groupby("continent")
            .agg(國家數=("iso_code", "size"),
                 最新接種率中位數=("vacc_rate", "median"),
                 最新完整接種率中位數=("full_rate", "median"))
            .join(d.groupby("continent")["deaths_per_million"].mean().rename("平均每日每百萬人死亡"))
            .round(2).sort_values("最新接種率中位數", ascending=False))
cont_sum.to_csv(RESULTS / "continent_summary.csv", encoding="utf-8-sig")
print(cont_sum)

# ---------- fig03 亞洲標竿 ----------
asia = latest[latest["continent"] == "亞洲"].copy()
asia["date"] = asia["date"].dt.strftime("%Y-%m-%d")
asia["人口(百萬)"] = asia["population"] / 1e6
asia = asia.sort_values("vacc_rate", ascending=True)
highlight = {"Singapore", "Japan", "Republic of Korea", "Israel", "Malaysia", "China"}
fig, ax = plt.subplots(figsize=(8, max(6, len(asia) * 0.28)))
colors = ["#d1495b" if c in highlight else "#bbb" for c in asia["country"]]
ax.barh(asia["country"], asia["vacc_rate"], color=colors)
ax.barh(asia["country"], asia["full_rate"], color="none", edgecolor="#333", linewidth=0.6)
ax.set_xlabel("接種率（%）；實心=至少一劑，框線=完整接種")
ax.set_title("亞洲各國最新接種率（紅色為標竿比較對象）")
fig.savefig(FIG / "fig03_asia_benchmark.png")
plt.close(fig)
bench = asia[["country", "人口(百萬)", "vacc_rate", "full_rate", "date"]].sort_values("vacc_rate", ascending=False)
bench.round(2).to_csv(RESULTS / "asia_benchmark.csv", index=False, encoding="utf-8-sig")

# 新加坡 vs 日本：接種速度
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, color in [("Singapore", "#d1495b"), ("Japan", "#30638e"), ("Republic of Korea", "#66a182"), ("Malaysia", "#edae49")]:
    g = d[d["country"] == name]
    if g.empty:
        continue
    ax[0].plot(g["date"], g["vacc_rate"], label=name, color=color)
    ax[1].plot(g["date"], g["deaths_per_million"].rolling(7, min_periods=1).mean(), label=name, color=color)
ax[0].set_title("至少一劑接種率（%）")
ax[1].set_title("每日每百萬人死亡數（7 筆移動平均）")
ax[0].legend(frameon=False)
for a in ax:
    a.tick_params(axis="x", rotation=45)
fig.suptitle("標竿比較：新加坡 vs 日本 vs 韓國 vs 馬來西亞", fontsize=14, y=1.03)
fig.savefig(FIG / "fig03b_benchmark_timeseries.png")
plt.close(fig)
print(bench.head(10).round(1).to_string(index=False))

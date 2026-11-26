"""
03 辛普森悖論：「接種率越高，死亡率越高？」

老師的提示：時間順序被打亂（把所有國家、所有日期混在一起，或只做橫斷面比較）時，
會出現「接種率越高、死亡率越高」的奇怪現象。

本程式的論證順序（v3）：
  A. 現象：同一個月份拿各國比較，相關係數多為正
  E. 主要證據（辛普森悖論）：同一個月份、依洲別分組後，歐洲內部轉為負相關
     → 同一時間點比較，沒有「時間趨勢」的干擾
  C. 可能原因：反向因果（先死得多 → 後打得快）或區域、經濟條件的干擾；分洲檢查
  F. 疫苗效果的較強證據：月份固定效果迴歸 + 以國家為單位的 bootstrap 信賴區間
  B、D. 補充：同一國家前後比較。限制：接種率只會上升，所以 B 等同「死亡 vs 時間」，
     無法分辨是疫苗效果還是疫情波段剛好過去

輸出：figures/fig04~fig06、results/simpson_*.csv、results/simpson_summary.txt
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from config import PROCESSED, FIG, RESULTS, CONTINENT_COLORS, RANDOM_STATE, setup_font

setup_font()
d = pd.read_csv(PROCESSED / "daily_clean.csv", parse_dates=["date"])
cm = pd.read_csv(PROCESSED / "country_month.csv")
out = []
SHOW_MONTH = "2021-05"      # fig04 示範用的月份
MIN_N_CONTINENT = 8         # 一個洲在某月至少要有幾國才計算相關

# ---------- A. 跨國比較（每個月份） ----------
rows = []
for m, g in cm.groupby("month"):
    if len(g) >= 20:
        r, p = spearmanr(g["vacc_rate"], g["deaths_pm"])
        rows.append({"month": m, "n_countries": len(g), "spearman_rho": r, "p_value": p})
between = pd.DataFrame(rows)
between.round(4).to_csv(RESULTS / "simpson_between_country_by_month.csv", index=False)
sig = between[between.p_value < 0.05]
out.append(f"A. 跨國比較（同月份、全部國家）：{(between.spearman_rho > 0).sum()}/{len(between)} 個月份相關係數為正，"
           f"其中 p < 0.05 的有 {len(sig)} 個月（{'、'.join(sig.month)}）；中位數 ρ = {between.spearman_rho.median():.3f}")

# ---------- E. 同月份、依洲別分組（辛普森悖論主要證據） ----------
rows = []
for (m, c), g in cm.groupby(["month", "continent"]):
    if len(g) >= MIN_N_CONTINENT:
        r, p = spearmanr(g["vacc_rate"], g["deaths_pm"])
        rows.append({"month": m, "continent": c, "n_countries": len(g), "spearman_rho": r, "p_value": p})
by_cont = pd.DataFrame(rows)
by_cont.round(4).to_csv(RESULTS / "simpson_by_month_continent.csv", index=False)
pivot = by_cont.pivot(index="month", columns="continent", values="spearman_rho") \
               .join(between.set_index("month")["spearman_rho"].rename("全部國家"))
pivot = pivot[["全部國家"] + [c for c in pivot.columns if c != "全部國家"]]
pivot.round(3).to_csv(RESULTS / "simpson_by_month_continent_pivot.csv", encoding="utf-8-sig")
eu = by_cont[by_cont.continent == "歐洲"]
out.append(f"E. 同月份分洲：歐洲內部 {(eu.spearman_rho < 0).sum()}/{len(eu)} 個月為負相關（中位數 ρ = {eu.spearman_rho.median():.3f}），"
           f"而全部國家一起看多為正相關 → 辛普森悖論")
for c, g in by_cont.groupby("continent"):
    out.append(f"   {c}：{len(g)} 個月可計算，中位數 ρ = {g.spearman_rho.median():.3f}")

# ---------- C. 反向因果或區域干擾（分洲檢查） ----------
early = d[d["date"] < "2021-03-01"].groupby("iso_code")["deaths_per_million"].mean().rename("early_deaths")
may = d[d["date"] <= "2021-05-31"].groupby("iso_code")["vacc_rate"].max().rename("vacc_by_may")
rc = pd.concat([early, may], axis=1).dropna().join(d.groupby("iso_code")[["country", "continent"]].first())
rows = []
for name, g in [("全部國家", rc), ("只看歐洲", rc[rc.continent == "歐洲"]), ("歐洲以外", rc[rc.continent != "歐洲"])]:
    r, p = spearmanr(g["early_deaths"], g["vacc_by_may"])
    rows.append({"範圍": name, "n": len(g), "spearman_rho": r, "p_value": p})
rc_tab = pd.DataFrame(rows)
rc_tab.round(4).to_csv(RESULTS / "simpson_reverse_causality.csv", index=False, encoding="utf-8-sig")
r_rc, p_rc = rc_tab.loc[0, ["spearman_rho", "p_value"]]
out.append("C. 早期死亡（2021/3 前）vs 5 月底接種率：" + "；".join(
    f"{r.範圍} n={r.n}, ρ={r.spearman_rho:.2f}, p={r.p_value:.3f}" for r in rc_tab.itertuples()))
out.append("   → 整體正相關有一部分來自洲別差異，應解讀為「反向因果或區域、經濟條件的干擾」，不是定論")

# ---------- F. 月份固定效果迴歸 ----------
# log(1+下月死亡) ~ 本月接種率 + log(1+本月死亡) + 月份虛擬變數 (+ 洲別虛擬變數)
# 月份虛擬變數 = 只拿「同一個月份」的國家互相比較，排除整體時間趨勢
fe = cm.dropna(subset=["deaths_pm_next"]).copy()
fe["y"] = np.log1p(fe["deaths_pm_next"])
fe["log_deaths"] = np.log1p(fe["deaths_pm"])


def fe_design(df, with_continent):
    parts = [df[["vacc_rate", "log_deaths"]], pd.get_dummies(df["month"], prefix="m", drop_first=True, dtype=float)]
    if with_continent:
        parts.append(pd.get_dummies(df["continent"], prefix="c", drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    X.insert(0, "const", 1.0)
    return X


def fe_coef(df, with_continent):
    X = fe_design(df, with_continent)
    beta, *_ = np.linalg.lstsq(X.values, df["y"].values, rcond=None)
    return beta[1]  # vacc_rate 的係數


rng = np.random.default_rng(RANDOM_STATE)
countries = fe["iso_code"].unique()
groups = {k: g for k, g in fe.groupby("iso_code")}
N_BOOT = 500
rows = []
for label, wc in [("控制月份", False), ("控制月份 + 洲別", True)]:
    b = fe_coef(fe, wc)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(countries, len(countries), replace=True)  # 以國家為單位抽樣，保留同國各月的相關性
        sample = pd.concat([groups[k] for k in pick], ignore_index=True)
        if sample["month"].nunique() == fe["month"].nunique():
            boots.append(fe_coef(sample, wc))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append({"模型": label, "n": len(fe), "接種率係數": b, "CI_low": lo, "CI_high": hi,
                 "接種率+10個百分點對下月死亡的影響(%)": (np.exp(10 * b) - 1) * 100,
                 "CI不含0": bool(hi < 0 or lo > 0)})
fe_tab = pd.DataFrame(rows)
fe_tab.round(4).to_csv(RESULTS / "simpson_month_fixed_effects.csv", index=False, encoding="utf-8-sig")
for r in fe_tab.itertuples():
    out.append(f"F. 月份固定效果迴歸（{r.模型}）：接種率每 +10 個百分點，下月死亡約 {r[6]:+.1f}%；"
               f"係數 95% CI [{r.CI_low:.4f}, {r.CI_high:.4f}]（國家為單位 bootstrap {N_BOOT} 次）")

# ---------- B. 國內比較（補充，有限制） ----------
rows = []
for iso, g in d.groupby("iso_code"):
    if len(g) >= 30 and g["vacc_rate"].nunique() > 5:
        r, p = spearmanr(g["vacc_rate"], g["deaths_per_million"])
        r_t, _ = spearmanr(g["date"].astype("int64"), g["deaths_per_million"])
        rows.append({"iso_code": iso, "country": g["country"].iat[0], "continent": g["continent"].iat[0],
                     "n_days": len(g), "spearman_rho": r, "p_value": p, "rho_date_vs_deaths": r_t})
within = pd.DataFrame(rows).dropna()
within.round(4).to_csv(RESULTS / "simpson_within_country.csv", index=False)
same = np.corrcoef(within.spearman_rho, within.rho_date_vs_deaths)[0, 1]
out.append(f"B.（補充）國內比較：{(within.spearman_rho < 0).mean():.0%} 的國家為負相關（中位數 ρ = {within.spearman_rho.median():.3f}）。"
           f"限制：接種率只會上升，與「日期 vs 死亡」的相關係數幾乎相同（兩者相關 = {same:.2f}），無法分辨疫苗效果與疫情波段")

# ---------- D. 突破 50% 前後（補充，有限制） ----------
rows = []
for iso, g in d.groupby("iso_code"):
    if g["vacc_rate"].min() < 50 <= g["vacc_rate"].max():
        rows.append({"iso_code": iso, "country": g["country"].iat[0], "continent": g["continent"].iat[0],
                     "before": g.loc[g.vacc_rate < 50, "deaths_per_million"].mean(),
                     "after": g.loc[g.vacc_rate >= 50, "deaths_per_million"].mean()})
ba = pd.DataFrame(rows)
ba.round(4).to_csv(RESULTS / "simpson_before_after_50pct.csv", index=False)
stat, p_ba = wilcoxon(ba["before"], ba["after"])
out.append(f"D.（補充）接種率突破 50% 前後：{len(ba)} 國中 {(ba.after < ba.before).mean():.0%} 死亡率下降；"
           f"中位數 {ba.before.median():.2f} → {ba.after.median():.2f}（Wilcoxon p = {p_ba:.2e}）。"
           f"限制：「前」包含 2020–21 冬季死亡高峰，同樣受時間影響")

# ==================== 圖 ====================
# fig04：單一月份，全部國家一條趨勢線 vs 各洲各一條
g = cm[cm["month"] == SHOW_MONTH]
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
xs = np.linspace(0, g["vacc_rate"].max(), 50)
for c, h in g.groupby("continent"):
    ax[0].scatter(h["vacc_rate"], h["deaths_pm"], s=28, alpha=.75, color=CONTINENT_COLORS[c], label=c)
    ax[1].scatter(h["vacc_rate"], h["deaths_pm"], s=28, alpha=.75, color=CONTINENT_COLORS[c], label=c)
    if len(h) >= MIN_N_CONTINENT:
        z = np.polyfit(h["vacc_rate"], h["deaths_pm"], 1)
        xx = np.linspace(h["vacc_rate"].min(), h["vacc_rate"].max(), 30)
        ax[1].plot(xx, np.polyval(z, xx), color=CONTINENT_COLORS[c], lw=2)
z = np.polyfit(g["vacc_rate"], g["deaths_pm"], 1)
ax[0].plot(xs, np.polyval(z, xs), "k--", lw=2, label="全部國家趨勢線")
r_all = between.set_index("month").loc[SHOW_MONTH, "spearman_rho"]
ax[0].set_title(f"全部國家一起看：正相關（ρ = {r_all:.2f}）")
eu_r = by_cont.set_index(["month", "continent"]).loc[(SHOW_MONTH, "歐洲"), "spearman_rho"]
ax[1].set_title(f"分洲看：歐洲內部反而是負相關（ρ = {eu_r:.2f}）")
for a in ax:
    a.set_xlabel("至少一劑接種率（%）")
ax[0].set_ylabel("每日每百萬人死亡數（月平均）")
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"辛普森悖論（{SHOW_MONTH}，同一個月份比較各國）", fontsize=14, y=1.03)
fig.savefig(FIG / "fig04_simpson_scatter.png")
plt.close(fig)

# fig05：每月相關係數 — 全部國家 vs 各洲內部
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(pivot.index, pivot["全部國家"], "k-o", lw=2.5, label="全部國家")
for c in [c for c in pivot.columns if c != "全部國家"]:
    ax.plot(pivot.index, pivot[c], "-o", ms=4, color=CONTINENT_COLORS[c], label=f"{c}內部")
ax.axhline(0, color="#999", lw=.8)
ax.set_ylabel("Spearman ρ（接種率 vs 死亡率）")
ax.set_title(f"每月跨國相關係數：全部國家多為正，歐洲內部多為負（各洲至少 {MIN_N_CONTINENT} 國才計算）")
ax.tick_params(axis="x", rotation=45)
ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
fig.savefig(FIG / "fig05_simpson_correlations.png")
plt.close(fig)

# fig06：左＝反向因果或區域干擾，右＝月份固定效果結果
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.3, 1], "wspace": .45})
for c, h in rc.groupby("continent"):
    ax[0].scatter(h["early_deaths"], h["vacc_by_may"], color=CONTINENT_COLORS[c], label=c, s=25)
for _, row in rc.nlargest(6, "early_deaths").iterrows():
    ax[0].annotate(row["country"], (row["early_deaths"], row["vacc_by_may"]), fontsize=8)
eu_rc = rc_tab.set_index("範圍").loc["只看歐洲"]
ax[0].set_xlabel("2021/3 以前 每日每百萬人死亡數")
ax[0].set_ylabel("2021/5 底 至少一劑接種率（%）")
ax[0].set_title(f"先死得多 → 後打得快？全部 ρ={r_rc:.2f}；只看歐洲 ρ={eu_rc.spearman_rho:.2f}（不顯著）", fontsize=11)
ax[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=2)
eff = fe_tab["接種率+10個百分點對下月死亡的影響(%)"]
lo = (np.exp(10 * fe_tab["CI_low"]) - 1) * 100
hi = (np.exp(10 * fe_tab["CI_high"]) - 1) * 100
ypos = np.arange(len(fe_tab))
ax[1].errorbar(eff, ypos, xerr=[eff - lo, hi - eff], fmt="o", color="#30638e", capsize=5, ms=8)
ax[1].axvline(0, color="#999", lw=.8)
ax[1].set_yticks(ypos, fe_tab["模型"])
ax[1].set_ylim(-.7, len(fe_tab) - .3)
ax[1].set_xlabel("接種率每 +10 個百分點，下月死亡變化（%）")
ax[1].set_title("月份固定效果迴歸（95% CI，國家 bootstrap）", fontsize=11)
fig.savefig(FIG / "fig06_reverse_causality_fixed_effects.png")
plt.close(fig)

(RESULTS / "simpson_summary.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))

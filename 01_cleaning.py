"""
01 資料清洗與特徵工程
輸入：data/raw/covid_vaccination_vs_death_ratio.csv
輸出：data/processed/daily_clean.csv、data/processed/country_month.csv、results/cleaning_log.txt
"""
import numpy as np
import pandas as pd

from config import RAW, PROCESSED, RESULTS, ISO_TO_CONTINENT

MIN_RECORDS = 15          # 紀錄筆數太少的國家，趨勢不可靠
MIN_POPULATION = 1_000_000  # 人口太少的國家，每百萬人死亡數波動極大（1 人死亡就會放大很多倍）

log = []
def note(msg):
    print(msg)
    log.append(msg)

df = pd.read_csv(RAW, index_col=0)
df["date"] = pd.to_datetime(df["date"])
note(f"原始資料：{len(df):,} 筆、{df['country'].nunique()} 個國家，日期 {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")
note(f"缺失值總數：{int(df.isna().sum().sum())}")

# 1. 重複列
dup = df.duplicated(["iso_code", "date"]).sum()
df = df.drop_duplicates(["iso_code", "date"])
note(f"[1] 重複（同國同日）資料：{dup} 筆，已移除")

# 2. 每日死亡數為負（各國事後修正前期數字造成）→ 設為 0
neg = (df["New_deaths"] < 0).sum()
note(f"[2] New_deaths 為負數：{neg} 筆（{', '.join(df.loc[df.New_deaths < 0, 'country'].unique())}），設為 0")
df["New_deaths"] = df["New_deaths"].clip(lower=0)

# 3. 接種比例 > 100%（Gibraltar：境外居民也在當地接種）→ 上限 100
over = (df["ratio"] > 100).sum()
note(f"[3] 接種比例超過 100%：{over} 筆（{', '.join(df.loc[df.ratio > 100, 'country'].unique())}），上限設為 100")

# 4. 重新計算比例（確認 ratio = people_vaccinated / population × 100）
df["vacc_rate"] = (df["people_vaccinated"] / df["population"] * 100).clip(upper=100)
df["full_rate"] = (df["people_fully_vaccinated"] / df["population"] * 100).clip(upper=100)
diff = (df["vacc_rate"] - df["ratio"].clip(upper=100)).abs().max()
note(f"[4] 驗證 ratio 欄位 = 至少一劑人數 / 人口 × 100，最大誤差 {diff:.2e}")

# 5. 完整接種人數不應大於至少一劑人數
bad = (df["people_fully_vaccinated"] > df["people_vaccinated"]).sum()
note(f"[5] 完整接種 > 至少一劑的不合理資料：{bad} 筆，以至少一劑人數為上限")
df["full_rate"] = np.minimum(df["full_rate"], df["vacc_rate"])

# 6. 每百萬人口死亡數（讓不同人口規模的國家可以比較）
df["deaths_per_million"] = df["New_deaths"] / df["population"] * 1e6

# 7. 洲別
df["continent"] = df["iso_code"].map(ISO_TO_CONTINENT)
assert df["continent"].notna().all(), df.loc[df.continent.isna(), "iso_code"].unique()
note(f"[7] 依 ISO 代碼加入洲別：{df.groupby('continent')['iso_code'].nunique().to_dict()}")

# 8. 過濾樣本太少或人口太少的國家
counts = df.groupby("iso_code").size()
pop = df.groupby("iso_code")["population"].first()
keep = counts[(counts >= MIN_RECORDS) & (pop >= MIN_POPULATION)].index
dropped = sorted(set(df["iso_code"]) - set(keep))
note(f"[8] 移除紀錄 < {MIN_RECORDS} 筆或人口 < {MIN_POPULATION:,} 的國家：{len(dropped)} 國")
df = df[df["iso_code"].isin(keep)].sort_values(["iso_code", "date"]).reset_index(drop=True)
note(f"清洗後：{len(df):,} 筆、{df['country'].nunique()} 個國家")

df.to_csv(PROCESSED / "daily_clean.csv", index=False)

# 9. 國家-月份彙總
# 原始資料各國的紀錄日期不連續（只有有接種更新的那天才有資料），
# 所以用「當月有紀錄日的平均每日死亡數」代替當月總和，避免紀錄天數多寡造成偏差。
df["month"] = df["date"].dt.to_period("M")
cm = (df.groupby(["iso_code", "country", "continent", "month"])
        .agg(deaths_pm=("deaths_per_million", "mean"),
             vacc_rate=("vacc_rate", "last"),
             full_rate=("full_rate", "last"),
             population=("population", "first"),
             n_days=("date", "size"))
        .reset_index())
cm = cm[cm["n_days"] >= 3]  # 一個月少於 3 天紀錄的月份不採用
cm = cm.sort_values(["iso_code", "month"])
g = cm.groupby("iso_code")
cm["deaths_pm_prev"] = g["deaths_pm"].shift(1)          # 上個月死亡
cm["vacc_rate_change"] = cm["vacc_rate"] - g["vacc_rate"].shift(1)  # 本月接種增加
cm["deaths_pm_next"] = g["deaths_pm"].shift(-1)          # 預測目標：下個月死亡
cm["month_idx"] = (cm["month"] - pd.Period("2020-12", "M")).apply(lambda x: x.n)
cm["month"] = cm["month"].astype(str)
cm.to_csv(PROCESSED / "country_month.csv", index=False)
note(f"[9] 國家-月份資料：{len(cm):,} 筆（每國每月一筆）")

(RESULTS / "cleaning_log.txt").write_text("\n".join(log), encoding="utf-8")

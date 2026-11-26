"""
05 預測模型：建模（結構優化、參數優化）→ 後處理／模型評估
任務：用「這個月」的接種與死亡狀況，預測「下個月」的死亡率
  - 迴歸：預測下個月每日每百萬人死亡數（取 log1p 降低極端值影響）
  - 分類：預測下個月死亡率是否上升（1=上升、0=持平或下降）

訓練集 / 測試集切分（依老師建議比較兩種）：
  ① 時間切：2021/07 以前的月份訓練、2021/08 以後測試（資料只到 2021/10，無法用 2021/12 切）
  ② 國家切：GroupKFold 5 折，每折用 80% 國家訓練、預測 20% 沒看過的國家

建模流程：
  1. 結構優化：比較不同模型（線性 / Ridge / 隨機森林 / 梯度提升），
     以「訓練集內部的交叉驗證成績」選出最終模型（不看測試集，避免偷看答案）
  2. 參數優化：GridSearchCV，只在訓練集內部做交叉驗證
       - 時間切：往前推的時間序列驗證（用 5 月以前預測 5 月、6 月以前預測 6 月、7 月以前預測 7 月）
       - 國家切：訓練集內再用 GroupKFold 4 折（同一國家不會同時出現在訓練與驗證）
  3. 後處理／評估：
       - 迴歸：預測值轉回每百萬人單位、殘差分析、預測最差的國家-月份、特徵重要性、消融實驗
       - 分類：混淆矩陣、閾值調整（在訓練集內部選出讓 F1 最高的閾值，再套到測試集）

輸出：results/model_*.csv、figures/fig09~fig13
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score,
                             root_mean_squared_error)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import PROCESSED, FIG, RESULTS, RANDOM_STATE, setup_font

setup_font()
cm = pd.read_csv(PROCESSED / "country_month.csv")
df = cm.dropna(subset=["deaths_pm_prev", "deaths_pm_next", "vacc_rate_change"]).copy()
df["log_deaths"] = np.log1p(df["deaths_pm"])
df["log_deaths_prev"] = np.log1p(df["deaths_pm_prev"])
df["y_reg"] = np.log1p(df["deaths_pm_next"])
df["y_clf"] = (df["deaths_pm_next"] > df["deaths_pm"]).astype(int)
print(f"建模資料：{len(df)} 筆（國家-月份），{df.iso_code.nunique()} 國；下月死亡上升比例 {df.y_clf.mean():.1%}")

NUM = ["log_deaths", "log_deaths_prev", "vacc_rate", "full_rate", "vacc_rate_change", "month_idx"]
CAT = ["continent"]
FEATURE_LABELS = {"log_deaths": "本月死亡", "log_deaths_prev": "上月死亡", "vacc_rate": "至少一劑接種率",
                  "full_rate": "完整接種率", "vacc_rate_change": "本月接種增加", "month_idx": "時間（月份）",
                  "continent": "洲別"}
TUNED, DEFAULT = "調參後", "未調參"


def prep(scale=True):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


# ---------- 候選模型：(建立模型的函式, 參數搜尋空間)；搜尋空間為 None 代表沒有可調參數 ----------
# 未調參 = 初版直接寫死的設定（Ridge 為 sklearn 預設 alpha=1），保留下來和調參後比較
REG_MODELS = {
    "Linear Regression": (lambda: Pipeline([("p", prep()), ("m", LinearRegression())]), None),
    "Ridge Regression": (lambda: Pipeline([("p", prep()), ("m", Ridge(alpha=1.0))]),
                         {"m__alpha": [0.1, 1, 10, 100]}),
    "Random Forest": (lambda: Pipeline([("p", prep(False)), ("m", RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=1))]),
        {"m__n_estimators": [200], "m__max_depth": [None, 6],
         "m__min_samples_leaf": [2, 5, 15], "m__max_features": [1.0, 0.5]}),
    "Gradient Boosting": (lambda: Pipeline([("p", prep(False)), ("m", GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE))]),
        {"m__n_estimators": [100, 300], "m__learning_rate": [0.03, 0.1],
         "m__max_depth": [2, 3], "m__subsample": [0.8, 1.0]}),
}
CLF_MODELS = {
    "Logistic Regression": (lambda: Pipeline([("p", prep()), ("m", LogisticRegression(max_iter=2000))]),
                            {"m__C": [0.01, 0.1, 1, 10]}),
    "Random Forest": (lambda: Pipeline([("p", prep(False)), ("m", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=1))]),
        {"m__n_estimators": [200], "m__max_depth": [None, 6],
         "m__min_samples_leaf": [2, 5, 15], "m__max_features": [1.0, 0.5]}),
}


def reg_metrics(y, pred):
    return {"MAE(log)": mean_absolute_error(y, pred), "RMSE(log)": root_mean_squared_error(y, pred),
            "R2": r2_score(y, pred),
            "MAE(每百萬人)": mean_absolute_error(np.expm1(y), np.expm1(pred))}


def clf_metrics(y, pred, prob):
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred),
            "AUC": roc_auc_score(y, prob) if len(set(y)) > 1 else np.nan}


def splits(kind):
    if kind == "時間切":
        tr = df["month"] <= "2021-07"
        yield df.index[tr], df.index[~tr]
    else:
        for tr, te in GroupKFold(n_splits=5).split(df, groups=df["iso_code"]):
            yield df.index[tr], df.index[te]


def inner_cv(train_idx, kind):
    """訓練集內部的驗證切分（回傳位置索引），只用來選模型與調參，完全不碰測試集。"""
    sub = df.loc[train_idx]
    if kind == "時間切":
        months = sorted(sub["month"].unique())
        return [(np.where(sub["month"] < m)[0], np.where(sub["month"] == m)[0]) for m in months[-3:]]
    return list(GroupKFold(n_splits=4).split(sub, groups=sub["iso_code"]))


def short(params):
    return json.dumps({k.replace("m__", ""): v for k, v in params.items()}, ensure_ascii=False)


def fit_candidates(models, X_tr, y_tr, cv, scoring):
    """回傳 {(模型, 版本): (已訓練模型, 內部驗證分數, 最佳參數)}"""
    out = {}
    for name, (make, grid) in models.items():
        default = make()
        cv_default = cross_val_score(default, X_tr, y_tr, cv=cv, scoring=scoring).mean()
        out[(name, DEFAULT if grid else "—")] = (default.fit(X_tr, y_tr), cv_default, "")
        if grid:
            gs = GridSearchCV(make(), grid, cv=cv, scoring=scoring, n_jobs=-1).fit(X_tr, y_tr)
            out[(name, TUNED)] = (gs.best_estimator_, gs.best_score_, short(gs.best_params_))
    return out


reg_rows, clf_rows, param_rows = [], [], []
final = {}   # 時間切的最終模型與預測，給後處理用
for kind in ["時間切", "國家切"]:
    for fold, (tr, te) in enumerate(splits(kind)):
        X_tr, X_te = df.loc[tr, NUM + CAT], df.loc[te, NUM + CAT]
        cv = inner_cv(tr, kind)

        # ----- 迴歸 -----
        reg_rows.append({"split": kind, "fold": fold, "model": "Baseline（下月=本月）", "version": "—",
                         "inner_cv_R2": np.nan, **reg_metrics(df.loc[te, "y_reg"], df.loc[te, "log_deaths"])})
        cands = fit_candidates(REG_MODELS, X_tr, df.loc[tr, "y_reg"], cv, "r2")
        for (name, ver), (m, score, params) in cands.items():
            reg_rows.append({"split": kind, "fold": fold, "model": name, "version": ver, "inner_cv_R2": score,
                             **reg_metrics(df.loc[te, "y_reg"], m.predict(X_te))})
            if params:
                param_rows.append({"task": "迴歸", "split": kind, "fold": fold, "model": name,
                                   "best_params": params, "inner_cv_score": round(score, 4)})
        # 結構優化：依「內部驗證分數」選最終模型
        best_key = max(cands, key=lambda k: cands[k][1])
        if kind == "時間切":
            final["reg"] = (best_key, cands[best_key][0], te)

        # ----- 分類 -----
        maj = int(df.loc[tr, "y_clf"].mean() >= .5)
        clf_rows.append({"split": kind, "fold": fold, "model": "Baseline（多數類別）", "version": "—",
                         "inner_cv_AUC": np.nan,
                         **clf_metrics(df.loc[te, "y_clf"], np.full(len(te), maj), np.full(len(te), .5))})
        trend = (df.loc[te, "deaths_pm"] > df.loc[te, "deaths_pm_prev"]).astype(int)
        clf_rows.append({"split": kind, "fold": fold, "model": "Baseline（延續上月趨勢）", "version": "—",
                         "inner_cv_AUC": np.nan, **clf_metrics(df.loc[te, "y_clf"], trend, trend)})
        cands = fit_candidates(CLF_MODELS, X_tr, df.loc[tr, "y_clf"], cv, "roc_auc")
        for (name, ver), (m, score, params) in cands.items():
            clf_rows.append({"split": kind, "fold": fold, "model": name, "version": ver, "inner_cv_AUC": score,
                             **clf_metrics(df.loc[te, "y_clf"], m.predict(X_te), m.predict_proba(X_te)[:, 1])})
            if params:
                param_rows.append({"task": "分類", "split": kind, "fold": fold, "model": name,
                                   "best_params": params, "inner_cv_score": round(score, 4)})
        best_key = max(cands, key=lambda k: cands[k][1])
        if kind == "時間切":
            final["clf"] = (best_key, cands[best_key][0], tr, te, cv)
    print(f"{kind} 完成")

KEYS = ["split", "model", "version"]
reg = pd.DataFrame(reg_rows).drop(columns="fold").groupby(KEYS, sort=False).mean().round(3)
clf = pd.DataFrame(clf_rows).drop(columns="fold").groupby(KEYS, sort=False).mean().round(3)
reg.to_csv(RESULTS / "model_regression_metrics.csv", encoding="utf-8-sig")
clf.to_csv(RESULTS / "model_classification_metrics.csv", encoding="utf-8-sig")
pd.DataFrame(param_rows).to_csv(RESULTS / "model_best_params.csv", index=False, encoding="utf-8-sig")
print("\n迴歸（國家切為 5 折平均）\n", reg.to_string())
print("\n分類（國家切為 5 折平均）\n", clf.to_string())

# 參數優化摘要：同一模型調參前後的測試成績
rows = []
for task, tab, metric in [("迴歸", reg, "R2"), ("分類", clf, "AUC")]:
    t = tab.reset_index()
    for (split, model), g in t.groupby(["split", "model"], sort=False):
        if set(g["version"]) >= {DEFAULT, TUNED}:
            p = pd.DataFrame(param_rows).query("task == @task and split == @split and model == @model")["best_params"]
            rows.append({"task": task, "split": split, "model": model, "metric": metric,
                         DEFAULT: g.loc[g.version == DEFAULT, metric].iat[0],
                         TUNED: g.loc[g.version == TUNED, metric].iat[0],
                         "最常選到的參數": p.mode().iat[0]})
tuning = pd.DataFrame(rows)
tuning["差異"] = (tuning[TUNED] - tuning[DEFAULT]).round(3)
tuning.to_csv(RESULTS / "model_tuning_summary.csv", index=False, encoding="utf-8-sig")
print("\n參數優化前後\n", tuning.to_string(index=False))

# 結構優化：依內部驗證分數選出的最終模型
sel = []
for task, tab, score in [("迴歸", reg, "inner_cv_R2"), ("分類", clf, "inner_cv_AUC")]:
    t = tab.reset_index().dropna(subset=[score])
    for split, g in t.groupby("split", sort=False):
        b = g.loc[g[score].idxmax()]
        sel.append({"task": task, "split": split, "selected": f"{b['model']}（{b['version']}）",
                    "inner_cv_score": b[score], "test_" + ("R2" if task == "迴歸" else "AUC"):
                        b["R2" if task == "迴歸" else "AUC"]})
sel = pd.DataFrame(sel)
sel.to_csv(RESULTS / "model_selection.csv", index=False, encoding="utf-8-sig")
print("\n依內部驗證選出的最終模型（國家切為各折平均分數最高者）\n", sel.to_string(index=False))


def label(model, version):
    return model if version == "—" else f"{model}（{version}）"


# ---------- fig09 模型比較（調參前後） ----------
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"wspace": .9})
for a, tab, metric, title in [(ax[0], reg, "R2", "迴歸：R²（越高越好）"),
                              (ax[1], clf, "AUC", "分類（下月死亡是否上升）：AUC")]:
    t = tab.reset_index()
    t["label"] = [label(m, v) for m, v in zip(t["model"], t["version"])]
    p = t.pivot(index="label", columns="split", values=metric).reindex(t["label"].unique()[::-1])
    p[["時間切", "國家切"]].plot.barh(ax=a, color=["#30638e", "#edae49"])
    a.set_title(title); a.set_ylabel("")
    a.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
ax[1].axvline(.5, color="#999", ls="--"); ax[1].set_xlim(.3, 1)
fig.savefig(FIG / "fig09_model_comparison.png")
plt.close(fig)

# ================= 後處理（時間切測試集） =================
(rname, rver), rmodel, te = final["reg"]
pred = rmodel.predict(df.loc[te, NUM + CAT])
r2_final = r2_score(df.loc[te, "y_reg"], pred)

# fig10 預測 vs 實際（轉回每百萬人）
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(np.expm1(df.loc[te, "y_reg"]), np.expm1(pred), s=14, alpha=.6, color="#30638e")
lim = max(np.expm1(df.loc[te, "y_reg"]).max(), np.expm1(pred).max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1)
ax.set_xlabel("實際 下月每日每百萬人死亡"); ax.set_ylabel("預測值")
ax.set_title(f"{label(rname, rver)}（時間切測試集）R²={r2_final:.2f}")
fig.savefig(FIG / "fig10_pred_vs_actual.png")
plt.close(fig)

# 殘差分析：哪些國家 / 月份預測最差
res = df.loc[te, ["country", "continent", "month", "deaths_pm", "deaths_pm_next"]].copy()
res["predicted"] = np.expm1(pred)
res["residual"] = res["deaths_pm_next"] - res["predicted"]          # 正 = 低估（實際比預測多）
res["abs_residual"] = res["residual"].abs()
res.sort_values("abs_residual", ascending=False).head(15).round(3) \
   .to_csv(RESULTS / "model_worst_predictions.csv", index=False, encoding="utf-8-sig")
by_cont = res.groupby("continent")["residual"].agg(["count", "mean", "median"]).round(3)
by_cont.to_csv(RESULTS / "model_residual_by_continent.csv", encoding="utf-8-sig")
print("\n殘差（實際 − 預測，每日每百萬人）依洲別\n", by_cont.to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].scatter(res["predicted"], res["residual"], s=14, alpha=.6, color="#30638e")
ax[0].axhline(0, color="k", lw=.8)
for i, (_, r) in enumerate(res.nlargest(5, "abs_residual").iterrows()):
    ax[0].annotate(f"{r['country']} {r['month']}", (r["predicted"], r["residual"]), fontsize=7,
                   xytext=(6, 8 if i % 2 == 0 else -10), textcoords="offset points")
ax[0].set_xlabel("預測值（每日每百萬人）"); ax[0].set_ylabel("殘差 = 實際 − 預測")
ax[0].set_title("殘差 vs 預測值（正值 = 低估）")
order = res.groupby("continent")["residual"].median().sort_values().index
ax[1].boxplot([res.loc[res.continent == c, "residual"] for c in order], showfliers=True)
ax[1].set_xticks(range(1, len(order) + 1), order)
ax[1].axhline(0, color="k", lw=.8)
ax[1].set_title("各洲殘差分布")
fig.suptitle(f"殘差分析：{label(rname, rver)}（時間切測試集）", fontsize=14, y=1.03)
fig.savefig(FIG / "fig13_residuals.png")
plt.close(fig)

# fig11 特徵重要性（Permutation importance）
imp = permutation_importance(rmodel, df.loc[te, NUM + CAT], df.loc[te, "y_reg"], n_repeats=30,
                             random_state=RANDOM_STATE, scoring="r2")
fi = pd.Series(imp.importances_mean, index=[FEATURE_LABELS[f] for f in NUM + CAT]).sort_values()
fi.round(4).to_csv(RESULTS / "model_feature_importance.csv", header=["importance_r2_drop"], encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(fi.index, fi.values, color="#00798c")
ax.set_xlabel("打亂該特徵後 R² 下降量")
ax.set_title(f"特徵重要性（{label(rname, rver)}）")
fig.savefig(FIG / "fig11_feature_importance.png")
plt.close(fig)
print("\n特徵重要性\n", fi.sort_values(ascending=False).round(3).to_string())

# 分類：閾值調整 + 混淆矩陣
(cname, cver), cmodel, tr, te, cv = final["clf"]
X_tr = df.loc[tr, NUM + CAT].reset_index(drop=True)
y_tr = df.loc[tr, "y_clf"].reset_index(drop=True)
oof_y, oof_p = [], []
for a, b in cv:  # 只用訓練集內部的驗證資料決定閾值
    m = clone(cmodel).fit(X_tr.iloc[a], y_tr.iloc[a])
    oof_y.append(y_tr.iloc[b]); oof_p.append(m.predict_proba(X_tr.iloc[b])[:, 1])
oof_y, oof_p = np.concatenate(oof_y), np.concatenate(oof_p)
thresholds = np.round(np.arange(0.2, 0.81, 0.05), 2)
f1s = [f1_score(oof_y, (oof_p >= t).astype(int)) for t in thresholds]
best_t = float(thresholds[int(np.argmax(f1s))])

y_te = df.loc[te, "y_clf"]
prob = cmodel.predict_proba(df.loc[te, NUM + CAT])[:, 1]
th_rows = []
for t in [0.5, best_t]:
    p = (prob >= t).astype(int)
    th_rows.append({"threshold": t, **clf_metrics(y_te, p, prob)})
th = pd.DataFrame(th_rows).round(3)
th.insert(0, "model", label(cname, cver))
th.to_csv(RESULTS / "model_threshold_tuning.csv", index=False, encoding="utf-8-sig")
print(f"\n閾值調整（{label(cname, cver)}，時間切；閾值由訓練集內部驗證選出 = {best_t}）\n", th.to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5), gridspec_kw={"width_ratios": [1.3, 1, 1], "wspace": .55})
ax[0].plot(thresholds, f1s, marker="o", color="#30638e")
ax[0].axvline(best_t, color="#d1495b", ls="--", label=f"選定閾值 {best_t}")
ax[0].axvline(0.5, color="#999", ls=":", label="預設 0.5")
ax[0].set_xlabel("閾值"); ax[0].set_ylabel("F1（訓練集內部驗證）")
ax[0].set_title("閾值調整"); ax[0].legend(frameon=False)
for a, t in zip(ax[1:], [0.5, best_t]):
    mat = confusion_matrix(y_te, (prob >= t).astype(int), labels=[0, 1])
    a.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, mat[i, j], ha="center", va="center", fontsize=13,
                   color="white" if mat[i, j] > mat.max() / 2 else "black")
    a.set_xticks([0, 1], ["預測：持平/下降", "預測：上升"])
    a.set_yticks([0, 1], ["實際：持平/下降", "實際：上升"])
    a.set_title(f"混淆矩陣（閾值 {t}）")
fig.suptitle(f"分類後處理：{label(cname, cver)}（時間切測試集）", fontsize=14, y=1.04)
fig.savefig(FIG / "fig12_confusion_threshold.png")
plt.close(fig)

# 線性迴歸係數（可解釋性）：控制本月死亡後，接種率對下月死亡的影響方向
lin = REG_MODELS["Linear Regression"][0]().fit(df[NUM + CAT], df["y_reg"])
coef = pd.Series(lin.named_steps["m"].coef_[:len(NUM)], index=[FEATURE_LABELS[f] for f in NUM])
coef.round(4).to_csv(RESULTS / "model_linear_coefficients.csv", header=["std_coef"], encoding="utf-8-sig")
print("\n線性迴歸標準化係數（全資料）\n", coef.round(3).to_string())

# ---------- 消融實驗：拿掉接種特徵，預測會變差多少？ ----------
VACC = ["vacc_rate", "full_rate", "vacc_rate_change"]
rows = []
for kind in ["時間切", "國家切"]:
    for use_vacc in [True, False]:
        num = NUM if use_vacc else [f for f in NUM if f not in VACC]
        scores = []
        for tr, te in splits(kind):
            pipe = Pipeline([("p", ColumnTransformer([("num", StandardScaler(), num),
                                                      ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])),
                             ("m", LinearRegression())])
            pipe.fit(df.loc[tr, num + CAT], df.loc[tr, "y_reg"])
            scores.append(r2_score(df.loc[te, "y_reg"], pipe.predict(df.loc[te, num + CAT])))
        rows.append({"split": kind, "features": "含接種特徵" if use_vacc else "不含接種特徵", "R2": np.mean(scores)})
abl = pd.DataFrame(rows).round(3)
abl.to_csv(RESULTS / "model_ablation_vaccination.csv", index=False, encoding="utf-8-sig")
print("\n消融實驗（Linear Regression）\n", abl.to_string(index=False))

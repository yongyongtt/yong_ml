"""
4. 적합성 평가 = 머신러닝

지역(리 72개 / 행정동 38개)마다 노선버스·똑버스·행복택시 적합도(0~100)를 낸다.

- 행복택시: 실제 운영 중인 마을(리 18개, 행정리 31개)을 정답으로 '방향 고정' 로지스틱 회귀 학습.
           각 조건의 방향을 상식대로 묶음 (예: 운행이 적을수록·고령이 많을수록 행복택시 쪽, 반대 방향 계수는 0).
           그냥 학습하면 겹치는 조건(대기시간↔운행 횟수, 걷는 시간↔500m 안 면적)끼리 부호가 뒤집혀
           "대기가 짧을수록 행복택시" 같은 설명이 나와서 막음. 정확도 0.755 → 0.729 (흔들림 ±0.021 → ±0.011).
           점수는 '그 마을을 빼고 학습한 모델'의 예측(교차검증)이라 운영 중인 마을도 공정하게 평가됨.
           + 택시 호출 많고 잘 안 잡히는 곳(앱택시, TAXI_BLEND 비율로 섞음)
- 똑버스:   정류장 단위로 '노선버스 1회당 승차'를 예측하는 모델 → 노선버스 효율이 안 나오는 곳
           + 정류장이 멀고(흩어짐) 밀도가 낮은 곳 + 택시 호출 많고 잘 안 잡히는 곳일수록 점수가 높음.
- 노선버스: 정류장 단위로 '비슷한 조건이면 하루 몇 회 다니는지' 예측하는 모델 → 실제가 예측보다 적은 곳(부족)
           + 붐비는 곳 + 밀도가 높은 곳 + 출퇴근 시간에 이동이 몰리는 곳일수록 점수가 높음.
- 앱택시·출퇴근 비중은 행정동 단위 데이터라 리는 그 면의 값을 같이 씀.
- 세 수단 모두 마지막에 민원(인구 1,000명당, 해당 수단 민원)을 COMPLAINT_BLEND 비율로 섞음.
- 실무 기준(하루 5회 이하 등)은 정답이 아니라 비교용으로 같이 표시.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from .features import pct

HAPPY_FEATURES = ["ratio70", "log_pop", "log_density", "log_access_runs", "access_wait_min", "walk_min_mean",
                  "cover500_share", "hub_time_min", "share_stops_le5"]
EFF_FEATURES = ["log_local_density", "ratio65", "fp_per_pop", "workers_per_pop", "log_dist_hub", "log_dist_rail",
                "hub_direct_n", "log_n_routes", "is_eupmyeon_n"]
SUP_FEATURES = ["log_local_density", "ratio65", "fp_per_pop", "workers_per_pop", "log_dist_hub", "log_dist_rail",
                "is_eupmyeon_n"]
LABEL = {
    "ratio70": "70세 이상 비율", "log_pop": "인구", "log_density": "인구밀도", "log_access_runs": "접근 정류장 운행 횟수",
    "access_wait_min": "버스 대기시간", "walk_min_mean": "정류장까지 도보 시간", "cover500_share": "정류장 500m 안 면적",
    "hub_time_min": "거점까지 시간", "share_stops_le5": "하루 5회 이하 정류장 비율",
}


# 행복택시 쪽으로 가는 방향: +1 = 클수록 / -1 = 작을수록 (이 방향과 반대인 계수는 0 으로 묶음)
HAPPY_SIGN = {"ratio70": 1, "log_pop": -1, "log_density": -1, "log_access_runs": -1, "access_wait_min": 1,
              "walk_min_mean": 1, "cover500_share": -1, "hub_time_min": 1, "share_stops_le5": 1}


class SignedLogit(ClassifierMixin, BaseEstimator):
    """방향 고정 로지스틱 회귀: 계수 부호를 signs 로 묶음 (L2 규제, 정답·오답 수 균형 가중)"""

    def __init__(self, signs=None, C=0.5):
        self.signs = signs
        self.C = C

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        n, p = X.shape
        w = np.where(y == 1, n / (2 * max(y.sum(), 1)), n / (2 * max(n - y.sum(), 1)))

        def f(t):
            b, c = t[0], t[1:]
            pr = 1 / (1 + np.exp(-(X @ c + b)))
            loss = -(w * (y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12))).sum() / n + c @ c / (2 * self.C * n)
            g = w * (pr - y)
            return loss, np.r_[g.sum() / n, X.T @ g / n + c / (self.C * n)]

        bounds = [(None, None)] + [(0, None) if s > 0 else (None, 0) for s in self.signs]
        r = minimize(f, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=bounds)
        self.intercept_, self.coef_ = np.array([r.x[0]]), r.x[1:][None, :]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(np.asarray(X, float) @ self.coef_[0] + self.intercept_[0])))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ================================================================ 모델 1. 행복택시 (리 단위 분류)
def happy_model(vil):
    X, y = vil[HAPPY_FEATURES].astype(float), vil["happy_taxi"].astype(int)

    def pipe():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             SignedLogit(signs=[HAPPY_SIGN[f] for f in HAPPY_FEATURES], C=0.5))

    # 교차검증 예측 (10번 반복 평균) → 점수로 씀
    oof = np.zeros(len(vil))
    aucs = []
    for rep in range(10):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED + rep)
        p = cross_val_predict(pipe(), X, y, cv=cv, method="predict_proba")[:, 1]
        oof += p / 10
        aucs.append(roc_auc_score(y, p))
    # 실무 기준(운행 횟수가 적을수록 행복택시)만으로 맞혔을 때
    runs = vil["access_runs"].fillna(vil["runs_max"])
    rule_auc = roc_auc_score(y, -runs.fillna(runs.max()))
    rule_hit = (vil["happy_taxi"] & (runs <= config.RULES["happy_runs"])).sum()
    full = pipe().fit(X, y)
    coef = full[-1].coef_[0]
    Z = full[:-1].transform(X)
    contrib = Z * coef  # (행, 특성) 기여도: 이 값이 클수록 행복택시 쪽으로 끌어올린 특성
    reasons, tops = [], []
    for i in range(len(vil)):
        top = [j for j in np.argsort(-contrib[i])[:3] if contrib[i, j] > 0]
        parts = [f"{LABEL[HAPPY_FEATURES[j]]} {_fmt(HAPPY_FEATURES[j], X.iloc[i, j])}"
                 f"(평균보다 {'높음' if Z[i, j] > 0 else '낮음'})" for j in top]
        reasons.append(" / ".join(parts))
        tops.append(",".join(HAPPY_FEATURES[j] for j in top))
    info = {
        "model": "happy_taxi_logistic_signed", "unit": "village", "target": "행복택시 운영 여부 (리 18개 / 72개)",
        "n": int(len(vil)), "n_positive": int(y.sum()), "features": HAPPY_FEATURES,
        "metrics": {"cv_auc_mean": round(float(np.mean(aucs)), 3), "cv_auc_std": round(float(np.std(aucs)), 3),
                    "rule_auc(운행 적을수록)": round(float(rule_auc), 3),
                    "rule_hit(운영 마을 중 하루 5회 이하)": f"{int(rule_hit)}/{int(y.sum())}"},
        "coef": {f: round(float(c), 3) for f, c in zip(HAPPY_FEATURES, coef)},
    }
    return oof, (reasons, tops), info


def _fmt(f, v):
    if pd.isna(v):
        return "-"
    if f in ("ratio70", "cover500_share", "share_stops_le5"):
        return f"{v * 100:.0f}%"
    if f in ("access_wait_min", "walk_min_mean", "hub_time_min"):
        return f"{v:.0f}분"
    if f == "log_access_runs":
        return f"하루 {np.expm1(v):.0f}회"
    if f == "log_pop":
        return f"{np.expm1(v):,.0f}명"
    if f == "log_density":
        return f"{np.expm1(v):,.0f}명/km²"
    return f"{v:.2f}"


# ================================================================ 모델 2·3. 정류장 단위 회귀 (똑버스 효율 / 노선버스 공급)
def _hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, random_state=config.SEED)


def stop_models(stp):
    out = {}
    infos = []
    # 2) 노선버스 1회당 승차 예측 (효율)
    m = stp[(stp["runs_wd"] >= 2) & stp["ride_per_run"].notna() & stp["dong_cd"].notna()]
    X, y, g = m[EFF_FEATURES].astype(float), np.log1p(m["ride_per_run"]), m["dong_cd"]
    oof = cross_val_predict(_hgb(), X, y, cv=GroupKFold(n_splits=5), groups=g)
    r2 = r2_score(y, oof)
    eff = _hgb().fit(X, y)
    out["eff_pred"] = np.expm1(eff.predict(stp[EFF_FEATURES].astype(float)))
    infos.append({"model": "bus_efficiency_hgb", "unit": "stop", "target": "1회 운행당 승차 (log)",
                  "n": int(len(m)), "features": EFF_FEATURES,
                  "metrics": {"cv_r2(동 단위로 나눔)": round(float(r2), 3)}})

    # 3) 하루 운행 횟수 예측 (비슷한 조건이면 몇 회 다니나)
    m = stp[(stp["runs_wd"] > 0) & stp["dong_cd"].notna()]
    X, y, g = m[SUP_FEATURES].astype(float), np.log1p(m["runs_wd"]), m["dong_cd"]
    oof = cross_val_predict(_hgb(), X, y, cv=GroupKFold(n_splits=5), groups=g)
    r2 = r2_score(y, oof)
    under = pd.Series(np.nan, index=stp.index)
    under.loc[m.index] = oof - y  # 양수 = 예측보다 적게 다님 (부족)
    out["undersupply"] = under
    infos.append({"model": "bus_supply_hgb", "unit": "stop", "target": "하루 운행 횟수 (log)",
                  "n": int(len(m)), "features": SUP_FEATURES,
                  "metrics": {"cv_r2(동 단위로 나눔)": round(float(r2), 3)}})
    stp = stp.assign(**out)
    return stp, infos


# ================================================================ 점수 만들기
# 점수는 '비교 집합' 안에서 백분위로 매긴다.
#  - 연간: 리 72개끼리 / 행정동 38개끼리
#  - 월별: 리 72개 × 12개월을 한꺼번에 (달마다 다시 매기지 않아야 월별 오르내림이 보임)
def _blend(model_score, complaint_rate):
    c = pct(complaint_rate).fillna(0)
    b = config.COMPLAINT_BLEND
    return ((1 - b) * model_score + b * c).round(1), c


def _reason_from_parts(parts: dict):
    """구성 요소 점수가 높은 순서로 이유 문장"""
    items = sorted([(k, v) for k, v in parts.items() if pd.notna(v)], key=lambda kv: -kv[1])[:2]
    return " / ".join(f"{k} ({'가장 높음' if v >= 99.5 else f'상위 {100 - v:.0f}%'})" for k, v in items if v >= 50)


def prepare_static(vil, reg, stp, happy_oof, happy_reasons):
    """모델 결과 중 지역마다 하나뿐인 값(행복택시 확률, 노선버스 예상 효율·부족 정도)을 리·동 표에 붙임"""
    s = stp[stp["li_cd"].notna()].groupby("li_cd")
    emd_eff = stp[stp["is_eupmyeon"]].groupby("dong_nm")["eff_pred"].median()
    vil["eff_pred"] = vil["li_cd"].map(s["eff_pred"].median()).fillna(vil["emd"].map(emd_eff))
    vil["undersupply"] = vil["li_cd"].map(s["undersupply"].median())
    vil["happy_prob"] = happy_oof
    vil["happy_reasons"], vil["happy_top"] = happy_reasons
    g = stp.groupby("dong_cd")
    reg["eff_pred"] = reg["dong_cd"].map(g["eff_pred"].median())
    reg["undersupply"] = reg["dong_cd"].map(g["undersupply"].median())
    return vil, reg


TAXI_LABEL = "택시 호출 많고 잘 안 잡힘"


def _taxi(F):
    """앱택시: 인구당 호출(시 평균 대비) + 미배차율. 둘 다 높을수록 '택시로도 채워지지 않는 이동'"""
    return pd.concat([pct(F["taxi_rel"]), pct(F["taxi_unmatched"])], axis=1).mean(axis=1)


def _drt(F):
    parts = pd.DataFrame({"노선버스 예상 효율 낮음": pct(F["eff_pred"], False),
                          "정류장까지 멂": pct(F["walk_min_mean"]),
                          "정류장 500m 밖 면적 큼": pct(F["cover500_share"], False),
                          "인구밀도 낮음": pct(F["density_km2"], False),
                          TAXI_LABEL: _taxi(F)})
    # 4가지를 같은 비중: 효율 / 멂(도보·500m) / 밀도 / 택시
    model = pd.concat([parts.iloc[:, 0], parts.iloc[:, 1:3].mean(axis=1), parts.iloc[:, 3], parts.iloc[:, 4]],
                      axis=1).mean(axis=1)
    return parts, model


def _bus(F, unit_word):
    parts = pd.DataFrame({f"비슷한 {unit_word}보다 운행 적음": pct(F["undersupply"]),
                          "버스가 붐빔": pct(F["ride_per_run"]),
                          "인구밀도 높음": pct(F["density_km2"]),
                          "출퇴근 시간에 이동이 몰림": pct(F["commute_share"])})
    return parts, parts.mean(axis=1)


HAPPY_LABEL = "지금 운영 마을과 조건이 비슷함"


def _happy(F):
    """행복택시 모델 점수 = 로지스틱 회귀 확률(백분위) + 앱택시(TAXI_BLEND 비율). 근거 문장에 택시도 붙임"""
    taxi = _taxi(F)
    tb = config.TAXI_BLEND
    parts = pd.DataFrame({HAPPY_LABEL: pct(F["happy_prob"]), TAXI_LABEL: taxi})
    model = ((1 - tb) * parts[HAPPY_LABEL] + tb * taxi).round(1)
    reasons = [r if t < 70 or pd.isna(t) else " / ".join(x for x in [r, f"{TAXI_LABEL} (상위 {100 - t:.0f}%)"] if x)
               for r, t in zip(F["happy_reasons"].fillna(""), taxi)]
    return parts, model, pd.Series(reasons, index=F.index)


def _parts_json(parts):
    """구성 요소 백분위 → JSON 문자열 (행마다). 화면 '자세히'·설명 문장 근거용"""
    if parts is None:
        return None
    return [json.dumps({k: round(float(v), 1) for k, v in r.items() if pd.notna(v)}, ensure_ascii=False)
            for r in parts.to_dict("records")]


def _rows(F, unit, region_cd, region_nm, items, rule, current):
    out = []
    for mode, score, model, cpt, reasons, parts in items:
        out.append(pd.DataFrame({
            "unit": unit, "period": F["period"].to_numpy(), "region_cd": region_cd.to_numpy(),
            "region_nm": region_nm.to_numpy(), "dong_cd": F["dong_cd"].to_numpy(), "dong_nm": F["dong_nm"].to_numpy(),
            "mvp": F["mvp"].to_numpy(), "mode": mode, "mode_nm": config.MODES[mode],
            "score": pd.Series(score).to_numpy(), "score_model": pd.Series(model).round(1).to_numpy(),
            "score_complaint": pd.Series(cpt).round(1).to_numpy(),
            "is_current": current[mode].to_numpy(), "rule_flag": rule[mode].fillna(False).to_numpy(),
            "top_reasons": pd.Series(reasons).to_numpy(),
            "parts": _parts_json(parts) or [None] * len(F),
        }))
    return out


def score_villages(F):
    """F: 리 표 (연간 72행 또는 월별 72×12행, period 컬럼)"""
    F = F.reset_index(drop=True)
    h_parts, h_model, h_reasons = _happy(F)
    h_score, h_c = _blend(h_model, F["cmp_HAPPY_TAXI_per1000"])
    d_parts, d_model = _drt(F)
    d_score, d_c = _blend(d_model, F["cmp_DRT_per1000"])
    b_parts, b_model = _bus(F, "마을")
    b_score, b_c = _blend(b_model, F["cmp_BUS_per1000"])
    runs = F["access_runs"].fillna(F["runs_max"])
    rule = {
        "HAPPY_TAXI": (runs <= config.RULES["happy_runs"]) | (F["runs_max"] <= config.RULES["happy_runs"]),
        "DRT": (F["density_km2"] < F["density_km2"].median()) & ((F["cover500_share"] < F["cover500_share"].median())
                                                                 | (F["walk_min_mean"] > F["walk_min_mean"].median())),
        "BUS": F["density_km2"] >= F["density_km2"].quantile(config.RULES["bus_density_pct"]),
    }
    current = {
        "BUS": F["n_stops"].fillna(0) > 0,
        "DRT": F.get("drt_zone", pd.Series(False, index=F.index)).map(lambda v: str(v).lower() in ("1", "true", "y")),
        "HAPPY_TAXI": F["happy_taxi"],
    }
    items = [
        ("HAPPY_TAXI", h_score, h_model, h_c, h_reasons, h_parts),
        ("DRT", d_score, d_model, d_c, d_parts.apply(lambda r: _reason_from_parts(r.to_dict()), axis=1), d_parts),
        ("BUS", b_score, b_model, b_c, b_parts.apply(lambda r: _reason_from_parts(r.to_dict()), axis=1), b_parts),
    ]
    return pd.concat(_rows(F, "village", F["li_cd"], F["emd"] + " " + F["li_nm"], items, rule, current),
                     ignore_index=True)


def score_dongs(G, vil_scores, vil):
    """G: 행정동 표 (연간 38행 또는 월별 38×12행). 행복택시는 같은 period 의 리 점수를 인구로 가중평균"""
    G = G.reset_index(drop=True)
    hv = vil_scores[vil_scores["mode"] == "HAPPY_TAXI"].merge(vil[["li_cd", "pop"]], left_on="region_cd", right_on="li_cd")
    h = hv.groupby(["dong_cd", "period"]).apply(lambda d: np.average(d["score"], weights=d["pop"].clip(lower=1)),
                                                include_groups=False)
    h_score = pd.Series([h.get((c, p), np.nan) for c, p in zip(G["dong_cd"], G["period"])]).round(1)
    h_reason = G["dong_nm"].map(lambda n: "읍면 마을들의 행복택시 점수(인구 가중)" if n in config.EUPMYEON
                                else "도시 지역이라 행복택시 대상 아님")
    d_parts, d_model = _drt(G)
    d_score, d_c = _blend(d_model, G["cmp_DRT_per1000"])
    b_parts, b_model = _bus(G, "동")
    b_score, b_c = _blend(b_model, G["cmp_BUS_per1000"])
    rule = {
        "HAPPY_TAXI": G["is_eupmyeon"] & (G["share_stops_le5"] >= config.RULES["happy_dong_share"]),
        "DRT": (G["density_km2"] < G["density_km2"].median()) & (G["cover500_share"] < G["cover500_share"].median()),
        "BUS": G["density_km2"] >= G["density_km2"].quantile(config.RULES["bus_density_pct"]),
    }
    current = {
        "BUS": G["n_stops"].fillna(0) > 0,
        "DRT": G.get("drt_zone", pd.Series(False, index=G.index)).map(lambda v: str(v).lower() in ("1", "true", "y")),
        "HAPPY_TAXI": G["happy_villages"].fillna(0) > 0,
    }
    items = [
        ("HAPPY_TAXI", h_score, h_score, pd.Series(np.nan, index=G.index), h_reason, None),
        ("DRT", d_score, d_model, d_c, d_parts.apply(lambda r: _reason_from_parts(r.to_dict()), axis=1), d_parts),
        ("BUS", b_score, b_model, b_c, b_parts.apply(lambda r: _reason_from_parts(r.to_dict()), axis=1), b_parts),
    ]
    return pd.concat(_rows(G, "dong", G["dong_cd"], G["dong_nm"], items, rule, current), ignore_index=True)


def finish(su):
    """가장 적합한 수단, AI 추천 여부, 실무 기준 비교, 지역 요약(부적합 정도) — period 별로"""
    su["ai_flag"] = su["score"] >= config.AI_RECOMMEND_SCORE
    su["rule_compare"] = np.select(
        [su["ai_flag"] & su["rule_flag"], su["ai_flag"] & ~su["rule_flag"], ~su["ai_flag"] & su["rule_flag"]],
        ["일치", "기준 밖 추가 발견", "기준만 해당"], "해당 없음")
    key = ["unit", "period", "region_cd"]
    su["rank_in_region"] = su.groupby(key)["score"].rank(ascending=False, method="first")
    su["is_best"] = su["rank_in_region"] == 1
    cur = su[su["is_current"]].groupby(key)["score"].max().rename("current_score")
    cur_modes = su[su["is_current"]].groupby(key)["mode"].apply(lambda m: ",".join(sorted(m))).rename("current_modes")
    best = su[su["is_best"]].set_index(key)[["mode", "score"]].rename(columns={"mode": "best_mode", "score": "best_score"})
    summ = best.join(cur).join(cur_modes).reset_index()
    summ["current_score"] = summ["current_score"].fillna(0)
    summ["current_modes"] = summ["current_modes"].fillna("")
    summ["mismatch"] = (summ["best_score"] - summ["current_score"]).clip(lower=0).round(1)
    cols = ["unit", "period", "region_cd", "region_nm", "dong_cd", "dong_nm", "mvp", "mode", "mode_nm", "score",
            "score_model", "score_complaint", "rank_in_region", "is_best", "is_current", "ai_flag", "rule_flag",
            "rule_compare", "top_reasons", "parts"]
    return su[cols], summ


def run(vil, reg, stp, vil_m, reg_m):
    """연간(2025) + 월별(202501~202512) 적합성 점수"""
    happy_oof, happy_reasons, info_h = happy_model(vil)
    stp, infos = stop_models(stp)
    vil, reg = prepare_static(vil, reg, stp, happy_oof, happy_reasons)
    static_v = vil[["li_cd", "eff_pred", "undersupply", "happy_prob", "happy_reasons", "happy_top"]]
    static_r = reg[["dong_cd", "eff_pred", "undersupply"]]
    vil_m = vil_m.merge(static_v, on="li_cd", how="left")
    reg_m = reg_m.merge(static_r, on="dong_cd", how="left")
    parts, summs = [], []
    for V, G in [(vil, reg), (vil_m, reg_m)]:   # 연간 / 월별 (각각 따로 백분위)
        sv = score_villages(V)
        sd = score_dongs(G, sv, vil)
        su, summ = finish(pd.concat([sv, sd], ignore_index=True))
        parts.append(su)
        summs.append(summ)
    su, summ = pd.concat(parts, ignore_index=True), pd.concat(summs, ignore_index=True)
    infos = [info_h] + infos
    return su, summ, infos, vil, reg, vil_m, reg_m, stp


def table(su):
    """ml_suitability: 지역 × 수단 점수만 (year·month 는 저장할 때 붙음). 문장·근거는 ml_explanation"""
    return su[["unit", "period", "region_cd", "mode", "score", "is_best", "is_current"]].reset_index(drop=True)


def model_info_table(infos, trained_at):
    """ml_model_info: 모델 이름, 무엇을 맞히는지, 학습 데이터 수, 정확도(JSON)"""
    return pd.DataFrame([{
        "model": i["model"], "target": i["target"], "n": i["n"],
        "metrics": json.dumps({**({"n_positive": i["n_positive"]} if i.get("n_positive") else {}), **i["metrics"]},
                              ensure_ascii=False),
        "trained_at": trained_at,
    } for i in infos])

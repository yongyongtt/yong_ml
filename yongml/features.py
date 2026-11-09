"""feat_ 표 → 모델·점수 계산에 쓰는 표 (리 / 행정동 / 정류장) + 민원 집계 + 월별 표"""
import numpy as np
import pandas as pd

import config


def pct(s, higher_is_more=True):
    """0~100 백분위 점수. higher_is_more=False 면 값이 작을수록 점수가 큼"""
    s = pd.to_numeric(s, errors="coerce")
    return ((s if higher_is_more else -s).rank(pct=True) * 100).round(1)


def complaint_counts(c, key, by_month=False):
    """민원 → 지역(key)별(월별) × 수단별 건수 (wide). index = key 또는 (key, ym)"""
    c = c[c[key].notna() & c["mode"].isin(config.MODES)].copy()
    idx = [key]
    if by_month:
        c["ym"] = pd.to_datetime(c["received_at"], errors="coerce").dt.strftime("%Y%m")
        c = c[c["ym"].str[:4] == config.YEAR]
        idx = [key, "ym"]
    w = c.groupby(idx + ["mode"]).size().unstack(fill_value=0)
    for m in config.MODES:
        if m not in w.columns:
            w[m] = 0
    w = w[list(config.MODES)].add_prefix("cmp_n_")
    w["cmp_n_all"] = w.sum(axis=1)
    return w


CMP_COLS = [f"cmp_n_{m}" for m in config.MODES] + ["cmp_n_all"] + \
           [f"cmp_{m}_per1000" for m in list(config.MODES) + ["all"]]


def add_complaints(df, keys, cnt):
    """민원 건수와 인구 1,000명당 건수를 붙임 (월별 표면 '그 달' 건수)"""
    df = df.drop(columns=[c for c in CMP_COLS if c in df.columns])
    df = df.merge(cnt, left_on=keys, right_index=True, how="left")
    for c in [c for c in df.columns if c.startswith("cmp_n_")]:
        df[c] = df[c].fillna(0).astype(int)
    pop = df["pop"].where(df["pop"] > 0)
    for m in list(config.MODES) + ["all"]:
        df[f"cmp_{m}_per1000"] = (df[f"cmp_n_{m}"] / pop * 1000).round(3)
    return df


COMMUTE_HOURS = [7, 8, 17, 18]   # 출근 7~9시, 퇴근 17~19시 (yong 03 의 첨두 시간과 같음)


def commute_share(hour_tbl):
    """행정동별 평일 '출퇴근 시간 생활인구 / 하루 평균' (클수록 출퇴근 이동이 몰리는 동)"""
    h = hour_tbl[hour_tbl["daytype"].astype(str) == "평일"].copy()
    h["hour"] = pd.to_numeric(h["hour"], errors="coerce")
    p = h.pivot_table(index="dong_cd", columns="hour", values="fp_avg", aggfunc="mean")
    return (p[COMMUTE_HOURS].mean(axis=1) / p.mean(axis=1)).round(3).rename("commute_share")


def taxi_monthly(rm, pop):
    """행정동 × 월 앱택시: 인구 1,000명당 호출(+시 평균 대비), 미배차율.
    달마다 출렁임이 커서 최근 3개월(그 달 포함)로 계산. 앱 이용이 해마다 늘어서 호출 수는 그 달 시 평균 대비로 비교"""
    t = rm[["dong_cd", "ym", "taxi_req", "taxi_matched"]].copy()
    for c in ("taxi_req", "taxi_matched"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.sort_values(["dong_cd", "ym"])
    g = t.groupby("dong_cd")
    req3 = g["taxi_req"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    mat3 = g["taxi_matched"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    n3 = g["taxi_req"].transform(lambda s: s.notna().astype(int).rolling(3, min_periods=1).sum())
    p = t["dong_cd"].map(pop).where(lambda x: x > 0)
    t["taxi_per1000"] = (req3 / n3.where(n3 > 0) / p * 1000).round(2)
    t["taxi_unmatched"] = (1 - mat3 / req3.where(req3 > 0)).round(3)
    t["taxi_match"] = (1 - t["taxi_unmatched"]).round(3)       # 성사율 (화면 그래프와 같은 말)
    t["taxi_rel"] = t["taxi_per1000"] / t["ym"].map(_city_mean(t, p, "ym"))
    return t[["dong_cd", "ym"] + TAXI_COLS]


def _city_mean(t, pop, by=None):
    """인구 가중 시 평균 (인구 1,000명당 호출)"""
    d = t.assign(_p=pop).dropna(subset=["taxi_per1000", "_p"])
    if by is None:
        return (d["taxi_per1000"] * d["_p"]).sum() / d["_p"].sum()
    return d.groupby(by).apply(lambda g: (g["taxi_per1000"] * g["_p"]).sum() / g["_p"].sum(), include_groups=False)


TAXI_COLS = ["taxi_per1000", "taxi_rel", "taxi_unmatched", "taxi_match"]
TAXI_COMMUTE = TAXI_COLS + ["commute_share"]


def build(T):
    reg = T["feat_region"].copy()
    vil = T["feat_village"].copy()
    stp = T["feat_stop"].copy()
    cmp_ = T["complaint"].copy()

    # ---- 리 (처인구 읍면 72개)
    emd2dong = dict(zip(reg["dong_nm"], reg["dong_cd"]))
    vil["dong_nm"] = vil["emd"]
    vil["dong_cd"] = vil["emd"].map(emd2dong)
    vil["mvp"] = vil["emd"].isin(config.MVP_EMD)
    vil["log_pop"] = np.log1p(vil["pop"])
    vil["log_density"] = np.log1p(vil["density_km2"])
    vil["log_access_runs"] = np.log1p(vil["access_runs"])
    vil["log_runs_max"] = np.log1p(vil["runs_max"])
    # 1회당 승차 = 리 안 정류장 일평균 승차 합 / 운행 횟수 합 (월별 표와 같은 정의)
    vil["ride_per_run"] = (vil["ride_daily"] / vil["runs_sum"].where(vil["runs_sum"] > 0)).round(3)
    vil["stops_per_km2"] = (vil["n_stops"] / vil["area_km2"].where(vil["area_km2"] > 0)).round(2)
    vil = add_complaints(vil, ["li_cd"], complaint_counts(cmp_, "li_cd"))
    vil["period"] = config.YEAR

    # ---- 행정동 38개
    # 앱택시(연간): 인구 1,000명당 월 호출, 미배차율 / 출퇴근 비중(평일 시간대별 생활인구)
    reg["taxi_per1000"] = pd.to_numeric(reg["taxi_req_per_1000pop_month"], errors="coerce")
    reg["taxi_unmatched"] = pd.to_numeric(reg["taxi_unmatched_rate"], errors="coerce")
    reg["taxi_match"] = (1 - reg["taxi_unmatched"]).round(3)
    reg["taxi_rel"] = (reg["taxi_per1000"] / _city_mean(reg, reg["pop"].where(reg["pop"] > 0))).round(3)  # 시 평균 = 1
    reg["commute_share"] = reg["dong_cd"].map(commute_share(T["feat_region_hour"]))
    # 리는 행정동(면) 값을 그대로 씀 (앱택시·생활인구가 행정동 단위 데이터라서)
    for c in TAXI_COMMUTE:
        vil[c] = vil["dong_cd"].map(reg.set_index("dong_cd")[c])
    reg["mvp"] = reg["dong_nm"].isin(config.MVP_EMD)
    reg["log_density"] = np.log1p(reg["density_km2"])
    reg["log_runs_median"] = np.log1p(reg["runs_median"])
    reg = add_complaints(reg, ["dong_cd"], complaint_counts(cmp_, "dong_cd"))
    reg["period"] = config.YEAR

    # ---- 정류장: 주변 맥락(동·리 지표)을 붙임
    ctx = reg.set_index("dong_cd")[["density_km2", "ratio65", "fp_per_pop", "workers_per_pop"]]
    stp = stp.merge(ctx, left_on="dong_cd", right_index=True, how="left")
    li_den = vil.set_index("li_cd")["density_km2"]
    stp["local_density"] = stp["li_cd"].map(li_den).fillna(stp["density_km2"])
    stp["log_local_density"] = np.log1p(stp["local_density"])
    stp["log_n_routes"] = np.log1p(stp["n_routes"])
    stp["log_dist_rail"] = np.log1p(stp["dist_rail_km"])
    stp["log_dist_hub"] = np.log1p(stp["dist_hub_km"])
    stp["is_eupmyeon_n"] = stp["is_eupmyeon"].astype(int)
    stp["hub_direct_n"] = stp["hub_direct"].astype(int)
    return vil, reg, stp


# ================================================================ 월별 표 (2025-01 ~ 12)
# 월마다 바뀌는 값만 갈아 끼우고, 나머지(인구·운행 횟수·도보 시간·모델 예측 등)는 연간 값 그대로 쓴다.
VIL_MONTHLY = ["ride_daily", "ride_per_run"]
REG_MONTHLY = ["ride_daily", "ride_per_run", "fp_avg", "taxi_unmatched_rate"]


def monthly(vil, reg, T):
    vm = T["feat_village_month"].copy()        # io_db 가 그 연도만 읽고 ym(YYYYMM) 을 만들어 둠
    v = vm[["li_cd", "ym"] + VIL_MONTHLY].merge(vil.drop(columns=VIL_MONTHLY + ["period"]), on="li_cd", how="inner")
    # 앱택시는 그 달 값(최근 3개월), 출퇴근 비중은 연간 값
    tm = taxi_monthly(T["feat_region_month"], reg.set_index("dong_cd")["pop"])
    v = v.drop(columns=TAXI_COLS).merge(tm, on=["dong_cd", "ym"], how="left")
    v = add_complaints(v, ["li_cd", "ym"], complaint_counts(T["complaint"], "li_cd", by_month=True))
    v["period"] = v["ym"]

    rm = T["feat_region_month"].copy()
    # 생활인구는 공공데이터 원본 그대로 씀 (평소와 크게 다른 달도 빼지 않음)
    r = rm[["dong_cd", "ym"] + REG_MONTHLY].merge(reg.drop(columns=REG_MONTHLY + ["period"]), on="dong_cd", how="inner")
    r = r.drop(columns=TAXI_COLS).merge(tm, on=["dong_cd", "ym"], how="left")
    r = add_complaints(r, ["dong_cd", "ym"], complaint_counts(T["complaint"], "dong_cd", by_month=True))
    r["period"] = r["ym"]
    return v.drop(columns=["ym"]), r.drop(columns=["ym"])

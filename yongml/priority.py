"""
5. 우선순위 평가 입력 점수

Spring 이 가중치로 합산할 수 있도록, 항목별 점수(0~100, 클수록 먼저 개선해야 함)를 미리 계산한다.
- 교통수단 부적합: 적합성 평가에서 '가장 적합한 수단 점수 - 지금 운영 중인 수단 점수'
- 접근성 취약: 정류장까지 도보 시간, 500m 밖 면적, 거점까지 시간
- 서비스 수준 부족: 운행 횟수(적을수록), 대기시간, 하루 5회 이하 정류장 비율
- 취약인구: 고령 비율(리: 60·70세 이상 / 동: 65·75세 이상, 장애인 비율)
- 민원: 인구 1,000명당 민원 수
- 잠재 수요: 인구, 생활인구, 승차
- 실행 가능성: 데이터가 없어 기본값 50 (시청 확인 후 반영)
score_total = 기본 가중치로 합산한 종합 취약도 (화면에서 가중치를 바꾸면 Spring 이 score_ 항목으로 다시 계산)
rank_total  = 같은 단위·같은 기간 안 순위 (1 = 가장 먼저 개선)
is_vulnerable = score_total 이 VULNERABLE_SCORE(50) 이상 → 대시보드 '교통 취약지역 N개'
대시보드 지도 보기: 종합 = score_total, 교통 수요 = score_demand, 공급 부족도 = score_service
ml_priority: year + month (연간 행은 month 빈 값, 월별 1~12). 월별에서 달마다 바뀌는 항목은 부적합·민원·잠재 수요,
             접근성·서비스 수준·취약인구는 연간 값 그대로(운행 시간표·인구가 한 시점뿐이라).
"""
import pandas as pd

import config
from .features import pct

COMPONENTS = ["mismatch", "access", "service", "vulnerable", "complaint", "demand", "feasibility"]


def _mean(*cols):
    return pd.concat(cols, axis=1).mean(axis=1).round(1)


def _village(V, gu):
    V = V.reset_index(drop=True)
    return pd.DataFrame({
        "unit": "village", "period": V["period"], "region_cd": V["li_cd"], "region_nm": V["emd"] + " " + V["li_nm"],
        "gu": V["dong_cd"].map(gu),
        "score_access": _mean(pct(V["walk_min_mean"]), pct(V["cover500_share"], False),
                              pct(V["hub_time_min"]), pct(V["access_dist_m"])),
        "score_service": _mean(pct(V["log_access_runs"], False), pct(V["access_wait_min"]),
                               pct(V["share_stops_le5"])),
        "score_vulnerable": _mean(pct(V["ratio70"]), pct(V["ratio60"])),
        "score_complaint": pct(V["cmp_all_per1000"]).fillna(0),
        "score_demand": _mean(pct(V["log_pop"]), pct(V["ride_daily"])),
    })


def _dong(G):
    G = G.reset_index(drop=True)
    return pd.DataFrame({
        "unit": "dong", "period": G["period"], "region_cd": G["dong_cd"], "region_nm": G["dong_nm"], "gu": G["gu"],
        "score_access": _mean(pct(G["walk_min_mean"]), pct(G["cover500_share"], False),
                              pct(G["hub_time_median"]), pct(G["dist_hub_km_mean"])),
        "score_service": _mean(pct(G["log_runs_median"], False), pct(G["wait_min_median"]),
                               pct(G["share_stops_le5"])),
        "score_vulnerable": _mean(pct(G["ratio65"]), pct(G["ratio75"]), pct(G["disabled_ratio"])),
        "score_complaint": pct(G["cmp_all_per1000"]).fillna(0),
        "score_demand": _mean(pct(G["fp_avg"]), pct(G["ride_daily"]), pct(G["pop"])),
    })


def run(vil, reg, vil_m, reg_m, summ):
    """연간 / 월별 각각 (월별은 12개월을 한꺼번에 백분위)"""
    gu = reg.set_index("dong_cd")["gu"]
    p = pd.concat([_village(vil, gu), _dong(reg), _village(vil_m, gu), _dong(reg_m)], ignore_index=True)
    p = p.merge(summ[["unit", "period", "region_cd", "mismatch"]], on=["unit", "period", "region_cd"], how="left")
    p["score_mismatch"] = p["mismatch"].fillna(0)   # 부적합 정도는 점수 차이 그대로(0~100)
    p["score_feasibility"] = 50.0
    w = config.PRIORITY_WEIGHTS
    p["score_total"] = (sum(p[f"score_{k}"].fillna(0) * w[k] for k in COMPONENTS) / sum(w.values())).round(1)
    p["rank_total"] = p.groupby(["unit", "period"])["score_total"].rank(ascending=False, method="min").astype(int)
    p["is_vulnerable"] = p["score_total"] >= config.VULNERABLE_SCORE
    cols = ["unit", "period", "region_cd", "region_nm", "gu"] + [f"score_{k}" for k in COMPONENTS] + \
           ["score_total", "rank_total", "is_vulnerable"]
    p = p[cols].sort_values(["unit", "period", "rank_total"]).reset_index(drop=True)
    return p

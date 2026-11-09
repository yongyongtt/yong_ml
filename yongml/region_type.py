"""
지역 유형 + 지표 신뢰도 → ml_region (지역마다 1행, 연간)

사각지대 유형 (6개 고정). 유형마다 정해진 지표의 백분위(그 유형 쪽 방향)를 평균해서 가장 높은 유형.
가장 높은 값이 TYPE_MIN 보다 낮으면 '대중교통 양호형'.
  외곽 대중교통 공백형  정류장 밀도↓ 운행↓ 거점까지 시간↑ 인구밀도↓   (행정동은 읍면만)
  버스 운행 부족형      운행↓ 대기↑ 하루 5회 이하 정류장 비율↑
  정류장 접근 취약형    정류장까지 도보↑ 500m 안 면적↓
  고령 이동 취약형      고령 비율↑(동 65세+·장애인, 리 70세+) 정류장까지 도보↑
  출퇴근 혼잡형        1회당 승차↑ 출퇴근 시간 생활인구↑ 인구밀도↑
  ※ 위 네 유형(공백·부족·접근·고령)은 접근성·서비스 전체가 중간보다 나쁜 곳만 (좋은 곳이 '공백형'으로 나오지 않게)
type_chips: 그 유형을 만든 지표 중 실제로 평균보다 그쪽이고 뚜렷한 것(백분위 CHIP_MIN 이상) 최대 3개 (양호형은 평균보다 좋은 지표)
  예) "정류장 밀도 적음 · 1.9곳/km² (평균 3.4곳/km²)"

지표 신뢰도 (표본 부족)
  행정동: 월평균 택시 호출 < RELIABILITY_MIN_CALLS
  리:     그 면의 월평균 택시 호출 < RELIABILITY_MIN_CALLS 또는 인구 < RELIABILITY_MIN_POP
  low_reliability = 참이면 reliability_note 에 화면에 띄울 한 줄
"""
import json

import numpy as np
import pandas as pd

import config
from .facts import chip, fact, fmt

TYPE_MIN = 60
CHIP_MIN = 60
GOOD = ("대중교통 양호형", "뚜렷한 공백이 없는 지역이에요")
# (유형, 설명, 행정동 지표, 리 지표) 지표 = [(컬럼, +1 클수록 / -1 작을수록 이 유형)]
TYPES = [
    ("외곽 대중교통 공백형", "정류장과 버스가 모두 적고 거점까지 먼 외곽 지역이에요",
     [("stops_per_km2", -1), ("runs_median", -1), ("hub_time_median", 1), ("density_km2", -1)],
     [("stops_per_km2", -1), ("access_runs", -1), ("hub_time_min", 1), ("density_km2", -1)]),
    ("버스 운행 부족형", "정류장은 있지만 버스가 드물게 다니는 지역이에요",
     [("runs_median", -1), ("wait_min_median", 1), ("share_stops_le5", 1)],
     [("access_runs", -1), ("access_wait_min", 1), ("share_stops_le5", 1)]),
    ("정류장 접근 취약형", "버스는 다니지만 정류장까지 멀어 걷기 힘든 지역이에요",
     [("walk_min_mean", 1), ("cover500_share", -1)],
     [("walk_min_mean", 1), ("cover500_share", -1)]),
    ("고령 이동 취약형", "어르신이 많은데 정류장까지 가기 어려운 지역이에요",
     [("ratio65", 1), ("disabled_ratio", 1), ("walk_min_mean", 1)],
     [("ratio70", 1), ("walk_min_mean", 1)]),
    ("출퇴근 혼잡형", "사람이 많아 출퇴근 시간에 버스가 붐비는 지역이에요",
     [("ride_per_run", 1), ("commute_share", 1), ("density_km2", 1)],
     [("ride_per_run", 1), ("commute_share", 1), ("density_km2", 1)]),
]
GAP_TYPES = [t[0] for t in TYPES[:4]]
NEED = {  # 접근성·서비스 전체 (나쁠수록 큼)
    "dong": [("walk_min_mean", 1), ("cover500_share", -1), ("hub_time_median", 1), ("runs_median", -1),
             ("wait_min_median", 1), ("share_stops_le5", 1)],
    "village": [("walk_min_mean", 1), ("cover500_share", -1), ("hub_time_min", 1), ("access_runs", -1),
                ("access_wait_min", 1), ("share_stops_le5", 1)],
}


def _p(df, col, d):
    """백분위(0~100). d=+1 클수록 / -1 작을수록 큼"""
    return (pd.to_numeric(df[col], errors="coerce") * d).rank(pct=True) * 100


def classify(df, unit):
    k = 2 if unit == "dong" else 3
    S = pd.DataFrame({t[0]: pd.concat([_p(df, c, d) for c, d in t[k]], axis=1).mean(axis=1) for t in TYPES})
    if unit == "dong":
        S.loc[~df["is_eupmyeon"].astype(bool).to_numpy(), "외곽 대중교통 공백형"] = np.nan
    need = pd.concat([_p(df, c, d) for c, d in NEED[unit]], axis=1).mean(axis=1)
    S.loc[(need < 50).to_numpy(), GAP_TYPES] = np.nan
    best = S.idxmax(axis=1, skipna=True)
    return best.where(S.max(axis=1) >= TYPE_MIN, GOOD[0]), S


def _chip_facts(unit, r, i, df, R, type_nm):
    """칩에 쓸 사실 (facts.fact 결과) 최대 3개"""
    spec = next((t[2 if unit == "dong" else 3] for t in TYPES if t[0] == type_nm), None)
    if spec is None:     # 양호형: 평균보다 좋은 접근성·서비스 지표
        spec = [(c, -d) for c, d in NEED[unit]]
    cand = []
    for k, (c, d) in enumerate(spec):
        f = fact(unit, r, R, c)
        if f and f["dir"] == d:
            cand.append((-float(_p(df, c, d).iloc[i]), k, f))
    cand.sort(key=lambda x: (x[0], x[1]))
    # 평균과 거의 같은 지표(그쪽 방향 백분위 CHIP_MIN 미만)는 빼고, 최소 1개는 남김
    return [f for k, (s, _, f) in enumerate(cand) if k == 0 or -s >= CHIP_MIN][:3]


def reliability(unit, r, calls):
    """(신뢰도 낮음 여부, 화면 한 줄, AI 카드 주의사항 한 줄)"""
    n = calls.get(r["dong_cd"], np.nan)
    low_calls = bool(pd.notna(n) and n < config.RELIABILITY_MIN_CALLS)
    low_pop = bool(unit == "village" and pd.notna(r["pop"]) and r["pop"] < config.RELIABILITY_MIN_POP)
    if not (low_calls or low_pop):
        return False, None, None
    nc, npop = fmt(n, "int"), fmt(r["pop"], "int")
    who = "이 마을은" if unit == "village" else "이 지역은"
    taxi = f"{'면 전체 ' if unit == 'village' else ''}월평균 택시 호출이 {nc}건"
    if low_pop and low_calls:
        note = f"{who} 인구가 {npop}명이고 {taxi}으로 적어 지표 신뢰도가 낮아요."
        caution = f"인구({npop}명)와 택시 호출(월평균 {nc}건)이 적어 추천 결과의 신뢰도가 낮아요."
    elif low_pop:
        note = f"{who} 인구가 {npop}명으로 적어 비율 지표의 신뢰도가 낮아요."
        caution = f"인구가 적어({npop}명) 비율 지표가 크게 흔들릴 수 있어 추천 결과의 신뢰도가 낮아요."
    else:
        note = f"{who} {taxi}으로 적어 지표 신뢰도가 낮아요."
        caution = f"택시 호출이 적어(월평균 {nc}건) 추천 결과의 신뢰도가 낮아요."
    return True, note + " 결과는 참고용으로만 봐 주세요.", caution


def run(vil, reg, R, region_month, label="연간"):
    """한 기간(연간 또는 한 달)의 지역 유형·지표 신뢰도
    vil·reg: 그 기간 값 (연간 표 또는 그 달 행). region_month: feat_region_month (연간이면 그 연도 전체, 월별이면 그 달)
      → 택시 호출: 연간은 월평균, 월별은 그 달 호출 수
    반환: (ml_region 표,
           {(unit, region_cd): AI 카드 주의사항 한 줄} - 신뢰도 낮은 곳만,
           {(unit, region_cd): 평균보다 나은 사실 목록} - 대중교통 양호형만 (현행 유지 근거))"""
    calls = pd.to_numeric(region_month["taxi_req"], errors="coerce").groupby(region_month["dong_cd"]).mean()
    rows, cautions, goods = [], {}, {}
    for unit, df, key in [("village", vil, "li_cd"), ("dong", reg, "dong_cd")]:
        df = df.reset_index(drop=True)
        types, _ = classify(df, unit)
        desc = dict([(t[0], t[1]) for t in TYPES] + [GOOD])
        for i, r in df.iterrows():
            low, note, caution = reliability(unit, r, calls)
            fs = _chip_facts(unit, r, i, df, R, types[i])
            rows.append({"unit": unit, "region_cd": r[key], "type_nm": types[i], "type_desc": desc[types[i]],
                         "type_chips": json.dumps([chip(f) for f in fs], ensure_ascii=False),
                         "low_reliability": low, "reliability_note": note})
            if types[i] == GOOD[0]:
                goods[(unit, r[key])] = fs
            if caution:
                cautions[(unit, r[key])] = caution
    out = pd.DataFrame(rows)
    return out, cautions, goods

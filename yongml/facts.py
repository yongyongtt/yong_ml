"""지표 한 개를 '이 지역 값 vs 평균' 사실로 만드는 도우미 (지역 유형 칩, AI 카드 근거 지표·문장에서 같이 씀)

비교 기준 = 평균(산술평균)
- 행정동: 용인시 행정동 38개 평균 → '용인시 평균'
- 리:     처인구 읍면 마을 72개 평균 → '읍면 마을 평균'. 택시·출퇴근은 면 값이라 행정동 평균과 비교 → '용인시 평균'
- 화면 그래프 '시 평균과 비교'에 나오는 지표(고령 비율, 택시 호출 성사율)는 그래프와 같은 숫자가 나오게
  yong 전처리의 feat_city / feat_city_month 값을 씀 (고령 비율 = 고령 인구 합 / 인구 합, 성사율 = 성사 합 / 호출 합)
"""
import re

import numpy as np
import pandas as pd

import config

# 지표: 컬럼 → (문장 주어, 칩 이름, 형식, 단위, 비교 집단, (클 때 말, 작을 때 말))
_BIG_SMALL = ("높음", "낮음")
_MANY = ("많음", "적음")
_LONG = ("김", "짧음")
SPEC = {
    "village": {
        "pop": ("인구", "인구", "int", "명", "self", _MANY),
        "ratio70": ("70세 이상 비율", "70세 이상 비율", "pct1", "", "self", _BIG_SMALL),
        "density_km2": ("인구밀도", "인구밀도", "int", "명/km²", "self", _BIG_SMALL),
        "stops_per_km2": ("정류장 밀도", "정류장 밀도", "f1", "곳/km²", "self", _MANY),
        "access_runs": ("가까운 정류장 하루 버스 운행", "하루 버스 운행", "int", "회", "self", _MANY),
        "access_wait_min": ("버스 예상 대기", "버스 대기", "int", "분", "self", _LONG),
        "walk_min_mean": ("정류장까지 걷는 시간", "정류장까지 도보", "int", "분", "self", _LONG),
        "cover500_share": ("정류장 500m 안에 드는 면적", "정류장 500m 안 면적", "pct", "", "self", ("넓음", "좁음")),
        "hub_time_min": ("거점(역·터미널)까지 걸리는 시간", "거점까지 시간", "int", "분", "self", _LONG),
        "share_stops_le5": ("하루 5회 이하 정류장 비율", "하루 5회 이하 정류장 비율", "pct", "", "self", _BIG_SMALL),
        "ride_per_run": ("버스 1회 운행당 승차", "버스 1회당 승차", "f2", "명", "self", _MANY),
        "eff_pred": ("비슷한 조건에서 예상되는 버스 1회 운행당 승차", "예상 승차", "f2", "명", "self", _MANY),
        "taxi_match": ("면 전체 택시 호출 성사율", "택시 호출 성사율(면)", "pct", "", "dong", _BIG_SMALL),
        "taxi_per1000": ("면 전체 인구 1,000명당 월 택시 호출", "택시 호출(면)", "int", "건", "dong", _MANY),
        "commute_share": ("면 전체 평일 출퇴근 시간 생활인구(하루 평균 대비)", "출퇴근 시간 생활인구(면)", "f2", "배",
                          "dong", _MANY),
    },
    "dong": {
        "pop": ("인구", "인구", "int", "명", "self", _MANY),
        "ratio65": ("65세 이상 비율", "65세 이상 비율", "pct1", "", "self", _BIG_SMALL),
        "disabled_ratio": ("장애인 비율", "장애인 비율", "pct1", "", "self", _BIG_SMALL),
        "density_km2": ("인구밀도", "인구밀도", "int", "명/km²", "self", _BIG_SMALL),
        "stops_per_km2": ("정류장 밀도", "정류장 밀도", "f1", "곳/km²", "self", _MANY),
        "runs_median": ("정류장 하루 버스 운행", "하루 버스 운행", "int", "회", "self", _MANY),
        "wait_min_median": ("버스 예상 대기", "버스 대기", "int", "분", "self", _LONG),
        "walk_min_mean": ("정류장까지 걷는 시간", "정류장까지 도보", "int", "분", "self", _LONG),
        "cover500_share": ("정류장 500m 안에 드는 면적", "정류장 500m 안 면적", "pct", "", "self", ("넓음", "좁음")),
        "hub_time_median": ("거점(역·터미널)까지 걸리는 시간", "거점까지 시간", "int", "분", "self", _LONG),
        "share_stops_le5": ("하루 5회 이하 정류장 비율", "하루 5회 이하 정류장 비율", "pct", "", "self", _BIG_SMALL),
        "ride_per_run": ("버스 1회 운행당 승차", "버스 1회당 승차", "f2", "명", "self", _MANY),
        "eff_pred": ("비슷한 조건에서 예상되는 버스 1회 운행당 승차", "예상 승차", "f2", "명", "self", _MANY),
        "taxi_match": ("택시 호출 성사율", "택시 호출 성사율", "pct", "", "self", _BIG_SMALL),
        "taxi_per1000": ("인구 1,000명당 월 택시 호출", "택시 호출", "int", "건", "self", _MANY),
        "commute_share": ("평일 출퇴근 시간 생활인구(하루 평균 대비)", "출퇴근 시간 생활인구", "f2", "배", "self", _MANY),
    },
}
GROUP_NM = {("village", "self"): "읍면 마을 평균", ("village", "dong"): "용인시 평균", ("dong", "self"): "용인시 평균"}
# 말 이어 쓰기: 표시 → (이어 쓰기, 끝맺음)
CONJ = {"높음": ("높아", "높아요"), "낮음": ("낮아", "낮아요"), "많음": ("많아", "많아요"), "적음": ("적어", "적어요"),
        "김": ("길어", "길어요"), "짧음": ("짧아", "짧아요"), "넓음": ("넓어", "넓어요"), "좁음": ("좁아", "좁아요")}


# ---------------------------------------------------------------- 글자 도우미
def fmt(v, kind):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    return {"pct": f"{v * 100:.0f}%", "pct1": f"{v * 100:.1f}%", "int": f"{v:,.0f}", "f1": f"{v:.1f}",
            "f2": f"{v:.2f}"}[kind]


def _num(s):
    return float(re.sub(r"[^\d.]", "", s) or 0)


def _jong(ch):
    """받침 번호 (없으면 0). 숫자는 읽는 소리 기준"""
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28
    return {"0": 21, "1": 8, "3": 16, "6": 1, "7": 8, "8": 8}.get(ch, 0)


def josa(word, a="이", b="가"):
    """받침 있으면 a, 없으면 b (예: 행복택시가 / 똑버스가)"""
    w = re.sub(r"\([^)]*\)$", "", word).strip()
    return a if w and _jong(w[-1]) else b


def ro(word):
    """으로 / 로 (받침 없거나 ㄹ 받침이면 '로')"""
    j = _jong(word[-1]) if word else 0
    return "로" if j in (0, 8) else "으로"


# ---------------------------------------------------------------- 비교 기준
def refs(vil, reg, city, city_m):
    """비교 기준 표 + 그래프 지표 평균(feat_city 값으로 덮어씀)"""
    c = city.copy()
    c["unit"] = c["unit"].astype(str)
    c = c.set_index("unit")
    req = pd.to_numeric(city_m["taxi_req"], errors="coerce").sum()
    mat = pd.to_numeric(city_m["taxi_matched"], errors="coerce").sum()
    over = {("dong", "ratio65"): float(c.loc["dong", "elderly_ratio"]),
            ("village", "ratio70"): float(c.loc["village", "elderly_ratio"])}
    if req > 0:
        over[("dong", "taxi_match")] = over[("village", "taxi_match")] = float(mat / req)
    return {"village": vil, "dong": reg, "over": over}


def fact(unit, r, R, col):
    """지표 하나: 이 지역 값, 비교 평균, 방향(+1 평균보다 큼 / -1 작음 / 0 같음). 값이 없으면 None
    strong = 평균과 FACT_MIN_GAP 이상 차이 (근거로 쓸 만함)"""
    spec = SPEC[unit].get(col)
    if not spec or col not in r.index:
        return None
    nm, short, kind, u, grp, words = spec
    ref = R["dong"] if grp == "dong" else R[unit]
    avg_v = R["over"].get((unit, col))
    if avg_v is None and col in ref.columns:
        avg_v = pd.to_numeric(ref[col], errors="coerce").mean()
    val, avg = fmt(r[col], kind), fmt(avg_v, kind)
    if val is None or avg is None:
        return None
    a, b = _num(val), _num(avg)      # 보이는 값(반올림)끼리 비교 → '10% vs 10% 높음' 같은 일 없음
    gap = abs(a - b) / abs(b) if b else float(a != 0)
    return {"col": col, "nm": nm, "short": short, "val": val, "avg": avg, "u": u, "grp": GROUP_NM[(unit, grp)],
            "grp_short": "평균" if grp == "self" else GROUP_NM[(unit, grp)], "dir": (a > b) - (a < b), "words": words,
            "strong": bool(a != b and gap >= config.FACT_MIN_GAP)}


def word(f):
    return f["words"][0] if f["dir"] > 0 else f["words"][1]


def fact_ctx(f):
    """LLM 에 넘기는 형태: 인구밀도 38명/km² (읍면 마을 평균 412명/km²보다 낮음). 차이가 작으면 '와 비슷'"""
    wa = josa(f"{f['avg']}{f['u']}", "과", "와")
    cmp = f"{wa} 같음" if f["dir"] == 0 else f"보다 {word(f)}" if f["strong"] else f"{wa} 비슷"
    return f"{f['nm']} {f['val']}{f['u']} ({f['grp']} {f['avg']}{f['u']}{cmp})"


def fact_sentence(f, end=True, short=False, numbers=True):
    """템플릿 문장: 인구밀도가 38명/km²로 읍면 마을 평균(412명/km²)보다 낮아요 / …낮아
    short=True 면 칩 이름과 '평균': 정류장까지 도보가 12분으로 평균(7분)보다 길어요
    numbers=False 면 숫자 없이 (카드 수단별 bullet 용, 숫자는 '근거 지표'에): 정류장까지 도보가 평균보다 길어요"""
    v = f"{f['val']}{f['u']}"
    nm, grp = (f["short"], f["grp_short"]) if short or not numbers else (f["nm"], f["grp"])
    head = f"{nm}{josa(nm)} {v}{ro(v)} {grp}({f['avg']}{f['u']})" if numbers else f"{nm}{josa(nm)} {grp}"
    if f["dir"] == 0:
        return head + josa(grp, "과", "와") + (" 같아요" if end else " 같아")
    return head + f"보다 {CONJ[word(f)][1 if end else 0]}"


def fact_line(f):
    """카드 '근거 지표' 한 줄: 택시 호출 성사율 45% (평균 55%)"""
    return f"{f['nm']} {f['val']}{f['u']} ({f['grp_short']} {f['avg']}{f['u']})"


def chip(f):
    """지역 유형 칩: 정류장 밀도 적음 · 1.9곳/km² (평균 3.4곳/km²)"""
    return f"{f['short']} {word(f)} · {f['val']}{f['u']} ({f['grp_short']} {f['avg']}{f['u']})"

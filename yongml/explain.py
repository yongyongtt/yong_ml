"""
7. 설명 문장 = LLM  → 화면 'AI 추천 해결방안' 카드 + 대시보드 TOP 10 한 줄

지역(리·행정동)마다 노선버스·똑버스·행복택시 3가지를 카드로 보여 줄 수 있게 만든다.
- 추천 1·2 / 제외는 서버(적합성 점수)가 정한다. LLM 은 정해진 결과를 문장으로만 바꾼다.
    1위 = 추천 1. 2위는 RECOMMEND_MIN 점 이상이고 1위와 RECOMMEND_GAP 점 안이면 추천 2, 아니면 제외.
- LLM 에는 점수·백분위 대신 '이 수단에 유리한 사실 / 불리한 사실'을 실제 값과 평균으로 넘긴다.
  (방향을 서버가 정해서 넘기므로 LLM 이 '높다/낮다'를 거꾸로 읽을 일이 없음)
- 숫자가 들어가는 '근거 지표' 줄은 서버가 만든다 (LLM 이 숫자를 잘못 옮길 일이 없음).
- 서버가 LLM 문장을 검사: 넘기지 않은 숫자, 해요체가 아닌 말투, 점수·순위·백분위 같은 말, 추천인데 '제외' 같은 모순,
  길이. 걸리면 이유를 알려 주고 다시 쓰게 하고(최대 3번), 그래도 안 되면 같은 모양의 템플릿 문장 (source = template).
- 지표 신뢰도가 낮은 지역(ml_region.low_reliability)은 그 주의사항을 서버가 맨 앞에 넣음.
- 현행 유지 (config.KEEP_CURRENT): 지역 유형이 '대중교통 양호형'이고 교통 취약지역이 아니면
  카드 맨 앞에 '현행 유지'(mode = KEEP)가 추천 1, 세 수단은 모두 제외 → 카드 4개. 아니면 세 수단 카드 3개.
  현행 유지 근거 = 그 지역이 평균보다 나은 접근성·서비스 지표 (ml_region.type_chips 와 같은 사실).

ml_explanation (지역 × 기간: year + month. 연간은 month 빈 값, 월별은 1~12)
  월별 카드는 그 달 점수·그 달 값(택시·승차·생활인구 등, 시 평균도 그 달)으로 만든다.
  이미 저장된 LLM 카드는 데이터(넘기는 사실·추천 결과)가 그대로면 다시 부르지 않고 그대로 쓴다 (--regen-llm 이면 새로).
  is_keep   현행 유지 지역이면 1 (카드 4개)
  options   JSON 배열, 카드 순서대로
            [{mode(BUS/DRT/HAPPY_TAXI/KEEP), name, subtitle, desc, status(추천/제외), rank(1·2, 제외는 null), score,
              is_current,
              reasons  그 수단 bullet 1~2개, 숫자 없이 (LLM. 추천 → 왜 맞는지 / 제외 → 왜 이번엔 덜 맞는지),
              facts    근거 지표 줄 (서버) 예) "택시 호출 성사율 45% (평균 55%)",
              together 추천이 2개일 때 '다른 추천 수단: 둘을 함께 하면 좋은 이유' 한 줄 (LLM), 없으면 null}]
  reasons   JSON 배열  이 지역에 추천 수단이 필요한 이유 1~3개 (숫자 + 기준 이름(값))
  cautions  JSON 배열  주의사항 0~3개
  summary              대시보드 미리보기 한 줄 (45자 이내, 명사로 끝남)
  문장 끝 마침표는 서버가 통일 (summary 는 마침표 없음). 평균과 FACT_MIN_GAP 안쪽으로 비슷한 값은 근거로 안 씀
  source               llm / template
  context              LLM 에 넘긴 사실 (확인용. '다시 생성' 때 prompts/explanation.md 에 이걸 넣어 다시 부르면 됨)
"""
import datetime as dt
import json
import re
import time

import pandas as pd
import requests

import config
from .facts import fact, fact_ctx, fact_line, fact_sentence, josa, ro, word
from .suitability import HAPPY_LABEL, HAPPY_SIGN, TAXI_LABEL

PROMPT = (config.ROOT / "prompts" / "explanation.md").read_text(encoding="utf-8")
# LLM_MODEL 을 비웠을 때 쓰는 모델. gemini 는 무료 등급 하루 한도가 넉넉한 Flash-Lite (Flash 는 하루 20회 정도라 부족)
DEFAULT_MODEL = {"gemini": "gemini-3.5-flash-lite", "anthropic": "claude-haiku-5-5"}
LIMIT = {"option": 60, "reason": 110, "caution": 90, "summary": 45, "together": 60}


class QuotaExceeded(Exception):
    """무료 등급 한도를 다 씀 → 이번 실행의 나머지 지역은 템플릿 문장으로"""


_LAST = [0.0]


def _wait():
    """분당 호출 수(LLM_RPM)를 넘지 않게 호출 사이를 띄움"""
    gap = 60.0 / max(config.LLM["rpm"], 1) - (time.time() - _LAST[0])
    if gap > 0:
        time.sleep(gap)
    _LAST[0] = time.time()


def _retry_delay(r):
    """429 응답에서 기다릴 초 (Retry-After 헤더 또는 Gemini 의 retryDelay '27s')"""
    try:
        if r.headers.get("Retry-After"):
            return float(r.headers["Retry-After"])
        for d in r.json().get("error", {}).get("details", []):
            if str(d.get("retryDelay", "")).endswith("s"):
                return float(d["retryDelay"][:-1])
    except (ValueError, AttributeError):
        pass
    return 30.0


STATE = {"quota": False}  # 하루 한도를 다 쓰면 True → 이번 실행의 나머지 기간·지역은 템플릿


def _post(url, headers, body):
    """POST
    - 429(한도): 분당 한도면 기다렸다 다시(최대 3번), 하루 한도면 QuotaExceeded
    - 5xx(503 과부하 등)·연결 끊김·시간 초과: config.LLM_RETRY_WAITS 초씩 기다렸다 다시. 다 써도 안 되면 오류 → 그 지역은 템플릿"""
    waits, n429, n5xx = list(config.LLM_RETRY_WAITS), 0, 0
    while True:
        _wait()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=90)
        except (requests.ConnectionError, requests.Timeout):
            if n5xx >= len(waits):
                raise
            time.sleep(waits[n5xx]); n5xx += 1
            continue
        if r.status_code == 429:
            if "PerDay" in r.text or "per day" in r.text.lower():
                raise QuotaExceeded("하루 한도를 다 씀 (태평양 시간 자정에 초기화)")
            if n429 >= 3:
                raise QuotaExceeded("분당 한도 초과가 계속됨")
            time.sleep(min(_retry_delay(r), 90)); n429 += 1
            continue
        if r.status_code >= 500 and n5xx < len(waits):
            time.sleep(waits[n5xx]); n5xx += 1
            continue
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------- 적합성 구성 요소(라벨) → 지표
OVERVIEW = {"village": ["pop", "ratio70"], "dong": ["pop", "ratio65", "disabled_ratio"]}
# 라벨 → [(지표, 이 수단에 유리한 방향: +1 클수록 / -1 작을수록)]
PART_SPEC = {
    "노선버스 예상 효율 낮음": [("eff_pred", -1)],
    "정류장까지 멂": [("walk_min_mean", 1)],
    "정류장 500m 밖 면적 큼": [("cover500_share", -1)],
    "인구밀도 낮음": [("density_km2", -1)],
    "인구밀도 높음": [("density_km2", 1)],
    TAXI_LABEL: [("taxi_match", -1), ("taxi_per1000", 1)],
    "버스가 붐빔": [("ride_per_run", 1)],
    "출퇴근 시간에 이동이 몰림": [("commute_share", 1)],
}
# 지표 하나로 말하기 어려운 모델 항목: (유리할 때, 불리할 때) 문장. '모델'이라는 말은 화면에 안 씀
MODEL_PART = {
    "비슷한 마을보다 운행 적음": ("비슷한 조건의 마을과 비교하면 버스 운행이 적은 편이에요",
                           "비슷한 조건의 마을과 비교해도 버스가 충분히 다니는 편이에요"),
    "비슷한 동보다 운행 적음": ("비슷한 조건의 지역과 비교하면 버스 운행이 적은 편이에요",
                          "비슷한 조건의 지역과 비교해도 버스가 충분히 다니는 편이에요"),
    HAPPY_LABEL: ("지금 행복택시가 다니는 마을들과 조건이 비슷해요", "지금 행복택시가 다니는 마을들과 조건이 달라요"),
}
# 템플릿 근거 뒷부분: 이 사실이 왜 이 수단에 맞는지
SO_WHAT = {
    "노선버스 예상 효율 낮음": "고정 노선보다 부를 때 가는 버스가 효율적이에요",
    "정류장까지 멂": "집 가까이 오는 수단이 필요해요",
    "정류장 500m 밖 면적 큼": "정류장에서 먼 곳까지 닿는 수단이 필요해요",
    "인구밀도 낮음": "이용이 흩어져 있어 부를 때 가는 수단이 맞아요",
    TAXI_LABEL: "택시만으로는 이동을 채우기 어려워요",
    "버스가 붐빔": "운행을 늘리면 효과가 커요",
    "인구밀도 높음": "고정 노선을 채울 수요가 있어요",
    "출퇴근 시간에 이동이 몰림": "정해진 시간 수요라 고정 노선 증차가 맞아요",
    HAPPY_LABEL: "집 앞에서 타는 행복택시가 도움이 돼요",
    "KEEP": "지금 대중교통으로도 이동 여건이 괜찮아요",
}
HAPPY_FEAT_FACT = {"ratio70": "ratio70", "log_pop": "pop", "log_density": "density_km2", "log_access_runs": "access_runs",
                   "access_wait_min": "access_wait_min", "walk_min_mean": "walk_min_mean",
                   "cover500_share": "cover500_share", "hub_time_min": "hub_time_min", "share_stops_le5": "share_stops_le5"}
NOT_TARGET = "도시 지역이라 행복택시 대상이 아니에요."
KEEP_NONE = "정류장 접근과 버스 운행이 모두 평균 수준이라 뚜렷한 공백이 없어요."


def _llm_target(o):
    """LLM 이 문장을 쓰는 카드 (도시 동의 행복택시는 서버 문장)"""
    return o["mode"] == "KEEP" or o["score"] is not None


# ---------------------------------------------------------------- 서버가 정하는 것: 추천/제외, 사실, 주의할 점
def decide(su_r, keep=False):
    """수단별 추천 1·2 / 제외 (점수 순). keep(현행 유지)이면 '현행 유지'가 추천 1, 세 수단은 모두 제외"""
    s = su_r.sort_values("score", ascending=False, na_position="last")
    top = s["score"].iloc[0]
    out, n_rec = [], 0
    if keep:
        name, sub, desc = config.MODE_CARD["KEEP"]
        out.append({"mode": "KEEP", "name": name, "subtitle": sub, "desc": desc, "status": "추천", "rank": 1,
                    "score": None, "is_current": False, "parts": {}, "reasons": [], "facts": [], "together": None})
        n_rec = 2      # 아래 세 수단은 모두 제외
    for r in s.itertuples():
        ok = pd.notna(r.score) and n_rec < 2 and (
            n_rec == 0 or (r.score >= config.RECOMMEND_MIN and top - r.score <= config.RECOMMEND_GAP))
        n_rec += int(ok)
        name, sub, desc = config.MODE_CARD[r.mode]
        out.append({"mode": r.mode, "name": name, "subtitle": sub, "desc": desc,
                    "status": "추천" if ok else "제외", "rank": n_rec if ok else None,
                    "score": None if pd.isna(r.score) else round(float(r.score), 1), "is_current": bool(r.is_current),
                    "parts": json.loads(r.parts) if isinstance(r.parts, str) else {}, "reasons": [], "facts": [],
                    "together": None})
    return out


def _taxi_facts(unit, r, R, want):
    """택시: '택시로도 못 채움'은 성사율이 평균보다 낮을 때만 유리. 호출이 많기만 한 건 근거가 아님"""
    ok, calls = fact(unit, r, R, "taxi_match"), fact(unit, r, R, "taxi_per1000")
    if not ok or not ok["strong"]:        # 평균과 거의 같으면 근거로 안 씀
        return []
    if want > 0 and ok["dir"] < 0:
        return [ok] + ([calls] if calls and calls["dir"] > 0 and calls["strong"] else [])
    if want < 0 and ok["dir"] > 0:
        return [ok]
    return []


def evidence(unit, o, r, R):
    """이 수단에 유리한 사실 / 불리한 사실. 구성 요소 점수 60 이상 / 40 이하 (유리한 게 2개 안 되면 50 이상까지),
    방향은 실제 값과 평균으로 다시 확인해서 맞는 것만. 평균과 FACT_MIN_GAP 안쪽으로 비슷한 값은 근거로 안 씀"""
    fav, unf = [], []
    strong = sum(v >= 60 for v in o["parts"].values())
    for label, v in o["parts"].items():
        if v >= 60 or (strong < 2 and v >= 50):
            side, want, order = fav, 1, -v
        elif v <= 40:
            side, want, order = unf, -1, v
        else:
            continue
        item = {"label": label, "order": order, "facts": [], "text": ""}
        if label in MODEL_PART:
            item["text"] = MODEL_PART[label][0 if want > 0 else 1]
            if label == HAPPY_LABEL and want > 0:   # 모델이 본 근거 지표 (행복택시 쪽 방향인 것만)
                keys = [k for k in str(r.get("happy_top") or "").split(",") if k in HAPPY_FEAT_FACT]
                fs = [(fact(unit, r, R, HAPPY_FEAT_FACT[k]), HAPPY_SIGN[k]) for k in keys]
                item["facts"] = [f for f, s in fs if f and f["dir"] == s and f["strong"]]
        elif label == TAXI_LABEL:
            item["facts"] = _taxi_facts(unit, r, R, want)
            if not item["facts"]:
                continue
        else:
            for col, d in PART_SPEC.get(label, []):
                f = fact(unit, r, R, col)
                if f and f["dir"] == d * want and f["strong"]:
                    item["facts"].append(f)
            if not item["facts"]:
                continue
        side.append(item)
    if o["mode"] == "HAPPY_TAXI" and unit == "dong" and not o["parts"] and o["score"] is not None:
        if o["score"] >= 60:
            fav.append({"label": "", "order": 0, "facts": [], "text": "면 안 마을들이 행복택시 조건에 잘 맞는 편이에요"})
        elif o["score"] <= 40:
            unf.append({"label": "", "order": 0, "facts": [], "text": "면 안 마을들이 행복택시 조건에 잘 맞지 않는 편이에요"})
    return sorted(fav, key=lambda x: x["order"]), sorted(unf, key=lambda x: x["order"])


def fact_lines(o, ev):
    """카드 '근거 지표' 줄 (서버가 만든 숫자): 추천은 유리한 사실 최대 3줄, 제외는 불리한 사실 최대 2줄"""
    fav, unf = ev
    items, n = (fav[:3], 3) if o["status"] == "추천" else (unf[:2], 2)
    lines = [fact_line(f) for it in items for f in it["facts"]]
    return list(dict.fromkeys(lines))[:n]


def caution_candidates(unit, r, opts, dummy, n_nocoord, used_cols):
    """주의할 점 후보 (서버가 데이터 상태를 보고 고름, 해요체). LLM 은 여기서만 골라 다듬음"""
    c = []
    if opts[0]["mode"] == "KEEP":
        alt = next((o for o in opts[1:] if o["score"] is not None), None)
        if alt:
            c.append(f"지금은 뚜렷한 공백이 없지만, 다시 검토한다면 {alt['name']}{josa(alt['name'])} 가장 가까운 대안이에요.")
    for o in [o for o in opts if o["status"] == "추천"]:
        if o["mode"] == "DRT":
            c.append("똑버스 실제 운영 구역 자료가 없어, 기존 운영 구역과 겹치는지 확인이 필요해요.")
        if o["mode"] == "HAPPY_TAXI" and o["is_current"]:
            c.append("이미 행복택시가 운영 중이라, 이용 실적을 보고 유지·확대를 판단해야 해요.")
        if o["mode"] == "BUS":
            c.append("하루 운행 횟수는 배차간격으로 계산한 추정치라, 실제 운행 기록(BMS) 확인이 필요해요.")
    if unit == "village" and used_cols & {"taxi_match", "taxi_per1000", "commute_share"}:
        c.append("택시·출퇴근 수치는 리가 아니라 면 전체 값이에요.")
    if n_nocoord:
        c.append(f"이 지역 정류장 {n_nocoord}곳은 위치 좌표가 없어 계산에서 빠졌어요.")
    if dummy:
        c.append("민원 수치는 시연용 데이터예요.")
    return list(dict.fromkeys(c))


def _item_ctx(it):
    parts = ([it["text"]] if it["text"] else []) + [fact_ctx(f) for f in it["facts"]]
    return "; ".join(parts)


def context(unit, region_nm, r, R, opts, ev, cautions, rule_compare, rtype, label=""):
    L = [f"[지역] {region_nm} ({'리' if unit == 'village' else '행정동'})"]
    if label:      # 월별 카드: 그 달 값·그 달 평균이라는 걸 알려 줌 (연간 카드는 이 줄 없음)
        L.append(f"[기준] {label} 값 (평균도 그 달 기준)")
    L += [f"[지역 유형] {rtype[0]} — {rtype[1]}", "", "[지역 개요]"]
    for col in OVERVIEW[unit]:
        f = fact(unit, r, R, col)
        if f:
            L.append(f"- {fact_ctx(f)}")
    cur = [o["name"] for o in opts if o["is_current"]]
    L.append(f"- 지금 운영 중인 수단: {', '.join(cur) if cur else '없음'}")
    if opts[0]["mode"] == "KEEP":
        pass                  # 현행 유지 지역은 실무 기준 비교(1위 수단 기준)를 넘기지 않음
    elif rule_compare == "기준 밖 추가 발견":
        L.append("- 실무 기준(하루 5회 이하 등)에는 안 걸리지만 AI 분석이 추가로 찾은 지역")
    elif rule_compare == "일치":
        L.append("- 실무 기준(하루 5회 이하 등)과도 일치")
    keep = opts[0]["mode"] == "KEEP"
    L += ["", "[수단별 결과 — 서버가 정함. 추천/제외를 바꾸지 말 것]"]
    for o in opts:
        if o["mode"] == "KEEP":
            L.append("■ KEEP 현행 유지(지금 수단 유지) → 추천 1 | 뚜렷한 교통 공백·혼잡이 없어 지금 수단을 그대로 둠")
            fav = ev["KEEP"][0]
            if fav:
                L.append("  현행 유지 근거 (평균보다 나은 점):")
                L += [f"   - {_item_ctx(it)}" for it in fav[:3]]
            continue
        tag = f"추천 {o['rank']}" if o["status"] == "추천" else ("제외 (현행 유지라서 바꾸지 않음)" if keep else "제외")
        L.append(f"■ {o['mode']} {o['name']}({o['subtitle']}) → {tag} | 지금 운영: {'예' if o['is_current'] else '아니오'}")
        if o["score"] is None:
            L.append("  - 도시 지역이라 행복택시 대상이 아님 (문장은 서버가 씀. 빈 배열로)")
            continue
        fav, unf = ev[o["mode"]]
        if keep:
            fav = []           # 현행 유지 지역은 바꾸지 않는 이유만
            if o["is_current"]:
                L.append(f"  - 지금 운영 중 → '지금 운영 중인 {o['name']}{ro(o['name'])} 충분해 늘리지 않아도 돼요' 관점으로")
            elif not unf:
                L.append("  - 불리한 사실 없음 → '지금 여건이 평균보다 나아 새로 도입할 필요가 크지 않아요' 관점으로")
        if fav:
            L.append("  이 수단에 유리한 사실:")
            L += [f"   - {_item_ctx(it)}" for it in fav[:3]]
        if unf:
            L.append("  이 수단에 불리한 사실:")
            L += [f"   - {_item_ctx(it)}" for it in unf[:2]]
    L += ["", "[주의할 점 후보]"] + ([f"- {c}" for c in cautions] or ["- (없음)"])
    return "\n".join(L)


# ---------------------------------------------------------------- 템플릿 (LLM 키가 없거나 실패했을 때, 같은 모양)
def _item_sentence(it, so_what=True, short=False, numbers=True):
    """근거 한 문장: 사실 + (왜 이 수단인지)"""
    if it["facts"]:
        tail = SO_WHAT.get(it["label"]) if so_what else None
        return fact_sentence(it["facts"][0], end=not tail, short=short, numbers=numbers) + (f", {tail}." if tail else ".")
    return it["text"] + "."


def _bullet(it, so_what=True):
    """카드 수단별 bullet: 숫자 없이 (숫자는 '근거 지표' 줄에)"""
    return _item_sentence(it, so_what=so_what, short=True, numbers=False)


def _action(o):
    if o["mode"] == "BUS":
        return "증차·노선 조정 검토"
    return "유지·확대 검토" if o["is_current"] else "도입 검토"


def template(opts, ev, cautions, F_over, rtype):
    """(수단별 bullet, 지역 근거, 주의사항, 한 줄 요약)"""
    rec = [o for o in opts if o["status"] == "추천"]
    keep = rec[0]["mode"] == "KEEP"
    bullets = {}
    for o in opts:
        fav, unf = ev[o["mode"]]
        if o["mode"] == "KEEP":
            b = [_bullet(it, so_what=False) for it in fav[:2]] or [KEEP_NONE]
        elif o["status"] == "추천":
            b = [_bullet(it) for it in fav[:2]] or \
                [f"{o['name']}{josa(o['name'])} 이 지역 조건에 가장 잘 맞아요."]
            if o["mode"] == "HAPPY_TAXI" and o["is_current"]:
                b = b[:1] + ["지금도 운영 중이라 유지·확대를 검토해요."]
        elif o["score"] is None:
            b = [NOT_TARGET]
        elif keep and o["is_current"]:
            b = [f"지금 운영 중인 {o['name']}{ro(o['name'])} 충분해 따로 늘리지 않아도 돼요."]
        elif keep:
            b = [fact_sentence(unf[0]["facts"][0], end=False, numbers=False) + ", 새로 도입할 필요가 크지 않아요."
                 if unf and unf[0]["facts"] else "지금 여건이 평균보다 나아 새로 도입할 필요가 크지 않아요."]
        elif unf and unf[0]["facts"]:
            b = [fact_sentence(unf[0]["facts"][0], end=False, numbers=False) + ", 이번엔 덜 맞아요."]
        elif unf:
            b = [unf[0]["text"] + "."]
        else:
            b = ["다른 수단보다 이 지역 조건에 덜 맞아요."]
        if o["status"] == "제외" and o["is_current"] and o["score"] is not None and not keep and len(b[0]) <= 50:
            b = ["지금 운영 중이지만 " + b[0]]
        bullets[o["mode"]] = b
    reasons, seen = [], set()
    for o in rec:
        for it in ev[o["mode"]][0]:
            s = _item_sentence(it)
            if s not in seen and len(reasons) < 3:
                reasons.append(s)
                seen.add(s)
    for f in ([] if keep else F_over):   # 모자라면 지역 개요에서 (도움이 되는 방향일 때만)
        if len(reasons) >= 3:
            break
        if f["col"] in ("ratio70", "ratio65", "disabled_ratio") and f["dir"] > 0 and f["strong"]:
            reasons.append(fact_sentence(f, end=False) + ", 이동이 불편한 주민을 위한 보완이 필요해요.")
    if not reasons:
        reasons = [KEEP_NONE] if keep else [f"{rec[0]['name']}{josa(rec[0]['name'])} 이 지역 조건에 가장 잘 맞아요."]
    first = next((f for it in ev[rec[0]["mode"]][0] for f in it["facts"]), None)
    head = f"{first['short']} {word(first)}" if first else rtype[0]
    summary = f"{head}, 현행 유지" if keep else f"{head}, {rec[0]['name']} {_action(rec[0])}"
    together = "함께 운영하면 서로 닿기 어려운 곳을 채워 줄 수 있어요." if len(rec) == 2 else None
    return bullets, reasons, cautions[:2], summary, together


# ---------------------------------------------------------------- LLM 결과 검사
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
BAD_END = re.compile(r"(습니다|입니다|됩니다|합니다|였다|이다|한다|함|임|됨|음)[.!]?$")
FORMAL_END = re.compile(r"(습니다|입니다|됩니다|합니다)[.!]?$")
BANNED = re.compile(r"(상위|하위) ?\d+ ?%|적합도|\d+ ?점(?![a-z가-힣])|순위|추천 ?[12]|단가|가정값|중앙값|백분위|모델")
# 카드 bullet 에 쓰면 안 되는 숫자 (숫자는 '근거 지표' 줄에 따로 나옴). 지표 이름 속 숫자는 빼고 봄
OPT_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(%|명|회|분|건|배|곳)")
NAME_NUM = re.compile(r"(하루 )?5회 이하|500 ?m|1,000명당|1회 운행당|1회당")
BARE_EXCLUDE = re.compile(r"^(새로 )?도입할 필요가 크지 않아요")
POINT = re.compile(r"평균 \d")


def _nums(s):
    out = set()
    for n in NUM.findall(s):
        try:
            out.add(f"{float(n.replace(',', '')):.2f}".rstrip("0").rstrip("."))
        except ValueError:
            pass
    return out


def _sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def _parse(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", t, re.S)
    return json.loads(m.group(0) if m else t)


def _strs(v):
    """문장 하나 또는 목록 → 앞뒤 공백 없앤 문장 목록"""
    v = [v] if isinstance(v, str) else (v if isinstance(v, list) else [])
    return [str(s).strip() for s in v if str(s).strip()]


def check(out, opts, ctx):
    """LLM 결과 검사. 문제를 모두 모아 한 줄로 (없으면 None)"""
    if not isinstance(out, dict):
        return "JSON 객체가 아님"
    o = out.get("options")
    targets = [x for x in opts if _llm_target(x)]
    if not isinstance(o, dict) or any(not _strs(o.get(x["mode"])) for x in targets):
        return "options 에 " + "·".join(x["mode"] for x in targets) + " 문장(목록)이 모두 있어야 함"
    rs, cs, sm = _strs(out.get("reasons")), _strs(out.get("cautions")), str(out.get("summary") or "").strip()
    if not 1 <= len(rs) <= 3 or len(cs) > 2 or not sm:
        return "reasons 는 1~3개, cautions 는 0~2개, summary 는 한 줄"
    probs = []
    n_rec = sum(x["status"] == "추천" for x in opts)
    tg = str(out.get("together") or "").strip()
    if n_rec == 2 and not tg:
        probs.append("추천이 2개라 together 한 문장이 필요")
    tg = tg if n_rec == 2 else ""
    opt_t = {x["mode"]: _strs(o[x["mode"]]) for x in targets}
    for x in targets:
        n_max = 2 if x["status"] == "추천" else 1
        if len(opt_t[x["mode"]]) > n_max:
            probs.append(f"{x['mode']} 문장은 {n_max}개까지")
    texts = [s for v in opt_t.values() for s in v] + rs + cs + ([tg] if tg else [])
    tone = [s for t in texts for s in _sentences(t) if BAD_END.search(s)]
    if tone:
        probs.append(f"해요체(~해요/~예요/~돼요)로 끝나지 않은 문장: {tone[:2]}")
    if re.search(r"요[.!]?$", sm) or FORMAL_END.search(sm):
        probs.append("summary 는 명사로 끝낼 것 (예: '… 도입 검토', '… 현행 유지')")
    banned = sorted({m.group(0) for t in [s for v in opt_t.values() for s in v] + rs + [sm, tg] for m in BANNED.finditer(t)})
    if banned:
        probs.append(f"쓰면 안 되는 말: {banned}")
    with_num = [s for v in opt_t.values() for s in v if OPT_NUM.search(NAME_NUM.sub("", s))]
    if with_num:
        probs.append(f"options 문장에는 숫자를 쓰지 말 것 (숫자는 근거 지표에 따로 나옴): {with_num[:2]}")
    if any(POINT.search(t) for t in rs + cs):
        probs.append("비교 값은 괄호로: '용인시 평균(54%)' 모양")
    if len(tg) > LIMIT["together"]:
        probs.append(f"together 가 김 ({LIMIT['together']}자 이내)")
    for x in targets:
        t = " ".join(opt_t[x["mode"]])
        if x["status"] == "추천" and re.search(r"제외|맞지 않|덜 맞", t):
            probs.append(f"{x['mode']} 는 추천인데 제외처럼 씀")
        if x["status"] == "제외" and re.search(r"월 예상|추천해", t):
            probs.append(f"{x['mode']} 는 제외인데 비용·추천 이야기를 씀")
        if x["status"] == "제외" and BARE_EXCLUDE.search(t):
            probs.append(f"{x['mode']} 제외 문장에 이유가 없음 ('지금 여건이 평균보다 나아 …'처럼 이유를 앞에)")
        if any(len(s) > LIMIT["option"] for s in opt_t[x["mode"]]):
            probs.append(f"{x['mode']} 문장이 김 ({LIMIT['option']}자 이내)")
    if any(len(t) > LIMIT["reason"] for t in rs) or any(len(t) > LIMIT["caution"] for t in cs):
        probs.append(f"근거는 {LIMIT['reason']}자, 주의사항은 {LIMIT['caution']}자 이내")
    if len(sm) > LIMIT["summary"]:
        probs.append(f"summary 가 김({len(sm)}자, {LIMIT['summary']}자 이내)")
    bad = _nums(" ".join(texts + [sm])) - _nums(ctx) - {"1", "2", "3"}
    if bad:
        probs.append(f"정보에 없는 숫자: {sorted(bad)} (계산해서 새 숫자를 만들지 말 것)")
    return " / ".join(probs) or None


def call_llm(prompt):
    prov, key = config.LLM["provider"], config.LLM["api_key"]
    model = config.LLM["model"] or DEFAULT_MODEL.get(prov, "")
    if not key or not model:
        return None, model
    if prov == "gemini":
        # 키는 주소가 아니라 헤더로 보냄 (로그·오류 메시지에 키가 남지 않게)
        d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                  {"x-goog-api-key": key, "content-type": "application/json"},
                  {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3,
                                        "maxOutputTokens": 4096}})
        parts = ((d.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip() or None, model
    if prov == "anthropic":
        d = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                  {"model": model, "max_tokens": 1200, "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in d.get("content", [])).strip(), model
    if prov == "openai":
        d = _post("https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {key}"},
                  {"model": model, "max_tokens": 1200, "messages": [{"role": "user", "content": prompt}]})
        return d["choices"][0]["message"]["content"].strip(), model
    raise ValueError(f"LLM_PROVIDER 를 알 수 없음: {prov} (gemini / anthropic / openai)")


def llm_card(ctx, opts, tries=3):
    """LLM 으로 카드 문장. 검사에 걸리면 문제를 알려 주고 다시 (최대 tries 번). (결과 또는 None, 실패 이유)"""
    prompt = PROMPT.replace("{context}", ctx)
    why = ""
    for _ in range(tries):
        text, _model = call_llm(prompt if not why else prompt + f"\n\n[앞 답변의 문제] {why}\n고쳐서 JSON 만 다시 써.")
        if text is None and not config.LLM["api_key"]:
            return None, "키 없음"
        if not text:
            why = "빈 답변"
            continue
        try:
            out = _parse(text)
        except ValueError:
            why = "JSON 형식이 아님"
            continue
        why = check(out, opts, ctx)
        if not why:
            return out, ""
    return None, why


# ---------------------------------------------------------------- 카드 완성 (LLM / 템플릿 공통)
def _dot(s):
    """문장 끝 마침표 통일"""
    s = str(s).strip()
    return s if not s or s[-1] in ".!?" else s + "."


def build_card(opts, ev, bullets, together=None):
    """수단별 bullet + 서버 근거 지표 + 함께 추천된 대안 → 카드 순서대로
    together: 두 추천 수단을 함께 하면 좋은 이유 한 문장 → 각 추천 카드에 '다른 수단 이름: 문장'"""
    for o in opts:
        o["reasons"] = [_dot(s) for s in bullets[o["mode"]]] if _llm_target(o) else [NOT_TARGET]
        o["facts"] = fact_lines(o, ev[o["mode"]]) if _llm_target(o) else []
    rec = [o for o in opts if o["status"] == "추천"]
    if len(rec) == 2 and together:
        for a, b in (rec, rec[::-1]):
            a["together"] = f"{b['name']}: {_dot(together)}"
    keys = ("mode", "name", "subtitle", "desc", "status", "rank", "score", "is_current", "reasons", "facts", "together")
    return [{k: o[k] for k in keys}
            for o in sorted(opts, key=lambda o: (o["status"] != "추천", o["rank"] or 9, -(o["score"] or -1)))]


# ---------------------------------------------------------------- 실행
def _reusable(old, ctx, opts, keep, rel):
    """저장된 LLM 카드를 그대로 써도 되는지: 넘기는 사실(context)·추천 결과(수단·상태·순서·점수)·현행 유지·신뢰도 문장이 같을 때만"""
    if old is None or old.get("source") != "llm" or old.get("context") != ctx or bool(old.get("is_keep")) != keep:
        return False
    try:
        was = {o["mode"]: (o["status"], o["rank"], o["score"]) for o in json.loads(old["options"])}
        caut = json.loads(old["cautions"])
    except (TypeError, ValueError, KeyError):
        return False
    if was != {o["mode"]: (o["status"], o["rank"], o["score"]) for o in opts}:
        return False
    old_rel = bool(caut) and "신뢰도가 낮아요" in caut[0]
    return (rel is None and not old_rel) or (rel is not None and bool(caut) and caut[0] == rel)


def run(vil, reg, su, R, region_tbl, rel_cautions, complaints, stp=None, use_llm=True, scope="all", goods=None, pri=None,
        period=None, label="연간", reuse=None, regen=False):
    """한 기간(연간 또는 한 달)의 카드
    vil·reg: 그 기간 값 (연간 표 또는 그 달 행), su·pri: 적합성·우선순위 (그 기간 행을 골라 씀)
    region_tbl: 그 기간 ml_region, rel_cautions: {(unit, region_cd): 지표 신뢰도 주의사항}
    goods: {(unit, region_cd): 평균보다 나은 사실} (대중교통 양호형)
    scope="mvp": 백암·원삼만 LLM, 나머지 지역은 템플릿 문장 (카드는 모든 지역)
    reuse: {(unit, region_cd): 저장된 카드 행} → 그대로 써도 되면 LLM 안 부름. regen=True 면 무시하고 새로"""
    period = period or config.YEAR
    vul = set()
    if pri is not None:
        a = pri[(pri["period"] == period) & pri["is_vulnerable"].astype(bool)]
        vul = set(zip(a["unit"], a["region_cd"]))
    keeps = {k: v for k, v in (goods or {}).items() if config.KEEP_CURRENT and k not in vul}
    dummy = bool(pd.to_numeric(complaints.get("is_dummy", pd.Series([0])), errors="coerce").fillna(0).astype(int).any())
    su = su[su["period"] == period]
    use_llm = use_llm and not STATE["quota"]
    nocoord = (stp[stp["lat"].isna()].groupby("dong_cd").size() if stp is not None and "lat" in stp.columns
               else pd.Series(dtype=int))
    rtypes = {(u, c): (n, d) for u, c, n, d in region_tbl[["unit", "region_cd", "type_nm", "type_desc"]].itertuples(index=False)}
    rows = []
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    for unit, base in [("village", vil), ("dong", reg)]:
        b = base.set_index("li_cd" if unit == "village" else "dong_cd")
        for cd, su_r in su[su["unit"] == unit].groupby("region_cd"):
            llm_here = use_llm and (scope == "all" or bool(su_r["mvp"].iloc[0]))   # mvp 밖은 템플릿
            r = b.loc[cd]
            rtype = rtypes.get((unit, cd), ("", ""))
            keep = (unit, cd) in keeps
            opts = decide(su_r, keep)
            ev = {o["mode"]: evidence(unit, o, r, R) for o in opts}
            if keep:
                good = [f for f in keeps[(unit, cd)] if f["strong"]]      # 평균과 비슷한 건 근거로 안 씀
                ev["KEEP"] = ([{"label": "KEEP", "order": i, "facts": [f], "text": ""} for i, f in enumerate(good)], [])
                for o in opts:      # 지금 운영 중인 수단은 '그대로 둠'만 (불리한 사실·근거 지표 안 붙임)
                    if o["mode"] != "KEEP" and o["is_current"]:
                        ev[o["mode"]] = ([], [])
            used = {f["col"] for m, (fav, unf) in ev.items()       # LLM 에 넘기는 사실에 쓴 지표
                    for it in (fav[:3] if not keep or m == "KEEP" else []) + unf[:2] for f in it["facts"]}
            cautions = caution_candidates(unit, r, opts, dummy, int(nocoord.get(cd, 0)) if unit == "dong" else 0, used)
            best = su_r[su_r["is_best"]]
            ctx = context(unit, su_r["region_nm"].iloc[0], r, R, opts, ev, cautions,
                          best["rule_compare"].iloc[0] if len(best) else "", rtype, "" if period == config.YEAR else label)
            old = (reuse or {}).get((unit, cd))
            if not regen and _reusable(old, ctx, opts, keep, rel_cautions.get((unit, cd))):
                rows.append({"unit": unit, "region_cd": cd, "is_keep": keep, "options": old["options"],
                             "reasons": old["reasons"], "cautions": old["cautions"], "summary": old["summary"],
                             "source": "llm", "created_at": old["created_at"], "context": ctx})
                continue
            out, src = None, "template"
            if llm_here:
                try:
                    out, _why = llm_card(ctx, opts)
                    if out:
                        src = "llm"
                except QuotaExceeded:
                    use_llm, STATE["quota"] = False, True
                except Exception:  # noqa: BLE001  (실패하면 템플릿 문장)
                    pass
            if out:
                bullets = {o["mode"]: _strs(out["options"].get(o["mode"])) for o in opts}
                reasons, caut, summary = _strs(out["reasons"]), _strs(out.get("cautions")), str(out["summary"]).strip()
                together = str(out.get("together") or "").strip() or None
            else:
                over = [f for f in (fact(unit, r, R, c) for c in OVERVIEW[unit]) if f]
                bullets, reasons, caut, summary, together = template(opts, ev, cautions, over, rtype)
            if (unit, cd) in rel_cautions:     # 지표 신뢰도 낮음 → 맨 앞
                caut = [rel_cautions[(unit, cd)]] + caut
            reasons, caut, summary = [_dot(s) for s in reasons], [_dot(s) for s in caut], summary.rstrip(". ")
            rows.append({
                "unit": unit, "region_cd": cd, "is_keep": keep,
                "options": json.dumps(build_card(opts, ev, bullets, together), ensure_ascii=False),
                "reasons": json.dumps(reasons, ensure_ascii=False), "cautions": json.dumps(caut[:3], ensure_ascii=False),
                "summary": summary, "source": src, "created_at": now, "context": ctx,
            })
    df = pd.DataFrame(rows)
    return df

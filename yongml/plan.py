"""통합 최적안 (A안): ML 결과에 비용·예산을 얹어 '어느 지역부터 무엇을 넣을지' 1·2·3안으로 정함
머신러닝·LLM 은 쓰지 않음. run_all.py 가 저장한 ml_ 결과와 feat_ 표를 읽어서 계산만 함

투입 수단  행정동별 AI 카드의 '추천 1' 그대로 (ml_explanation). 서버가 정한 추천·제외를 바꾸지 않음
           (동 지역은 AI 카드에서 행복택시가 이미 제외라서 추천 1 로 나오지 않음)
후보       종합 취약도 순위(rank_total) TOP CANDIDATE_TOP(15) 안의 행정동 중 (같은 점수는 같이 포함)
           (대시보드 '교통 취약지역'(50점 이상)과는 따로. 50점 기준은 외곽 읍·면만 남아 수단이 똑버스 하나뿐이라
            15위까지 넓혀 외곽 위주는 유지하면서 노선버스도 들어올 수 있게)
           - 추천 1 이 '현행 유지'면 제외
           - 추천 1 이 똑버스인데 ML 결과가 '지금 운영: 예'면 새로 드는 돈이 없어 제외
             (노선버스는 증차, 행복택시는 운영 중이면 실제 지원금으로 계산하므로 제외하지 않음)
비용       2개월 = 대수 × 1대 1일 운영비 × DAYS. 행복택시는 지원금
           - 대수: 행정동 면적 3등분. 똑버스 하위 2·가운데 3·상위 4대 / 노선버스 하위·가운데 1·상위 2대
           - 노선버스 단가: 그 행정동 정류장을 가장 많이 지나는 노선(주 노선)이 마을버스면 마을버스, 아니면 시내버스
             (노선 원본 표 raw_gg_route_info·raw_gg_route_stations 를 못 읽으면 전부 시내버스 단가)
           - 행복택시: 운영 중인 읍·면은 그 안 리들의 월평균 지원금 합 × 2, 새로 도입하면 운영 중인 읍·면 평균
           - 차량비는 운송원가·위탁비에 들어 있어 계산에서 빼고, 지금 버스가 없는 곳의 노선버스만 참고 문구
효과       종합 취약도 × 65세 이상 인구 (인구 전체로 하면 도심 동이 유리해져서 고령 인구로)
예산       BUDGET. 후보 비용 합계가 예산보다 작으면 합계의 절반
1·2·3안    계산은 같고 줄 세우는 기준만 다름
           1안 효율형: 효과 ÷ 비용 높은 순 / 2안 형평형: 종합 취약도 높은 순
           3안 균형형: 두 순위의 평균이 작은 순 (같으면 종합 취약도 높은 쪽)
           앞에서부터 예산이 찰 때까지 담고, 남은 예산보다 비싼 곳은 건너뛰고 다음으로 ('예산 밖')
리(里)     소속 읍·면(행정동) 결과를 그대로 (리 단위 화면용)
"""
import json

import numpy as np
import pandas as pd

import config
from . import io_db
from .errors import JobError

# ============================== 단가·예산·대수 기준 (바꾸면 run_plan.py 를 다시 돌리면 됨) ==============================
CANDIDATE_TOP = 15                 # 후보: 행정동 종합 취약도 순위 이 안 (같은 점수는 같이 포함)
DAYS = 61                          # 2개월
COST_DAY = {                       # 1대 1일 운영비 (원)
    "CITY_BUS": 809_181,           # 시내버스: 광주광역시 2025 대형 1일 대당 표준운송원가 (경기도 미공개라 대용)
    #                                https://www.gwangju.go.kr/traffic/contentsView.do?pageId=traffic34
    "VILLAGE_BUS": 585_726,        # 마을버스: 경기도 평균 (2024)
    "DRT": 500_000,                # 똑버스: 용인 2026 사업비 1대 연 1억 8,250만 원 ÷ 365
}
DRT_YEAR_BUDGET = 1_825_000_000    # 용인 똑버스 2026 사업비 (3개 읍)
# 예산(가정): 읍당 연 사업비(÷3) → 처인구 읍·면 7곳으로 확대 → 2개월. 약 7.1억
BUDGET = round(DRT_YEAR_BUDGET / 3 * len(config.EUPMYEON) * DAYS / 365)
UNITS = {"DRT": (2, 3, 4), "BUS": (1, 1, 2)}   # 행정동 면적 하위 1/3, 가운데, 상위 1/3
HAPPY_MONTHS = 2                               # 행복택시: 월평균 지원금 × 2
VEHICLE_NOTE = {"CITY_BUS": "차량 구입 시 약 3.5억 (계산에서 제외, 참고)",
                "VILLAGE_BUS": "차량 구입 시 약 0.75억~0.94억 (계산에서 제외, 참고)"}
PLANS = {   # 번호: (이름, 한 줄 설명, 줄 세우는 기준, 순위 이름)
    1: ("효율형", "같은 예산으로 더 많은 사람에게 효과가 가도록", "효과 ÷ 비용 높은 순", "효율"),
    2: ("형평형", "비용이 들어도 가장 취약한 곳부터", "종합 취약도 높은 순", "취약도"),
    3: ("균형형", "효율과 취약도를 함께 고려한 절충안", "효율 순위와 취약도 순위의 평균", "균형"),
}
# =====================================================================================================================

BUS_NM = {"CITY_BUS": "시내버스", "VILLAGE_BUS": "마을버스"}


def _eok(won):
    """원 → '1.2억' (1억 미만도 억으로: 0.6억)"""
    return f"{won / 1e8:.1f}억"


def _man(won):
    return f"{won / 1e4:,.1f}만 원".replace(".0만", "만")


# ---------------------------------------------------------------- 입력
def _feat(eng, name, year):
    return io_db._tidy(io_db._pick_year(name, io_db._read(eng, name), year))


def _ml(name, year):
    df = io_db.read_saved(name, year)
    if df.empty:
        raise JobError("E_NO_DATA", f"{name} {year}년 없음 (run_all.py 먼저)")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    return df


def _ids(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def bus_kind(eng, stp):
    """행정동별 주 노선(그 동 정류장을 가장 많이 지나는 노선)의 종류. {dong_cd: ("VILLAGE_BUS"|"CITY_BUS", 노선명)}
    노선 원본 표를 못 읽으면 {} → 전부 시내버스 단가 (보수적)"""
    try:
        ri = io_db._read(eng, "raw_gg_route_info")
        rs = io_db._read(eng, "raw_gg_route_stations")
        lk = pd.DataFrame({"routeId": _ids(rs["routeId"]), "stop_id": _ids(rs["stationId"])}).drop_duplicates()
        lk = lk.merge(stp[["stop_id", "dong_cd"]].dropna(), on="stop_id")
        n = lk.groupby(["dong_cd", "routeId"]).size().rename("n").reset_index()
        main = n.sort_values(["dong_cd", "n", "routeId"], ascending=[True, False, True]).drop_duplicates("dong_cd")
        rid = _ids(ri["routeId"])
        typ = dict(zip(rid, _ids(ri["routeTypeCd"])))
        name = dict(zip(rid, ri["routeName"].astype("string"))) if "routeName" in ri.columns else {}
    except Exception:  # noqa: BLE001  (표가 없거나 형식이 다르면 시내버스 단가로)
        return {}
    return {d: ("VILLAGE_BUS" if typ.get(r) == "30" else "CITY_BUS", name.get(r))
            for d, r in zip(main["dong_cd"], main["routeId"])}


def load(year):
    eng = io_db.connect()
    reg = _feat(eng, "feat_region", year)
    vil = _feat(eng, "feat_village", year)
    stp = _feat(eng, "feat_stop", year)
    kinds = bus_kind(eng, stp)
    eng.dispose()
    return {"reg": reg, "vil": vil, "kinds": kinds,
            "pri": _ml("ml_priority", year), "exp": _ml("ml_explanation", year), "su": _ml("ml_suitability", year)}


# ---------------------------------------------------------------- AI 카드 추천 1
def _top(row):
    """AI 카드 options 에서 추천 1 → (mode, 지금 운영 여부). 'KEEP' = 현행 유지"""
    try:
        opts = json.loads(row) if isinstance(row, str) else []
    except ValueError:
        opts = []
    for o in opts:
        if o.get("status") == "추천" and o.get("rank") == 1:
            return o.get("mode"), bool(o.get("is_current"))
    return None, False


def picks(exp, su, month):
    """행정동별 추천 1. AI 카드(ml_explanation)가 기준이고, 카드가 없는 지역만 적합성 1위(ml_suitability)로"""
    sel = exp["month"].isna() if pd.isna(month) else exp["month"] == month
    e = exp[sel & (exp["unit"] == "dong")]
    out = {cd: _top(o) for cd, o in zip(e["region_cd"].astype(str), e["options"])}
    sel = su["month"].isna() if pd.isna(month) else su["month"] == month
    s = su[sel & (su["unit"] == "dong") & su["is_best"].astype(bool)]
    for cd, m, cur in zip(s["region_cd"].astype(str), s["mode"], s["is_current"]):
        if out.get(cd, (None, False))[0] is None:
            out[cd] = (m, bool(cur))
    return out


# ---------------------------------------------------------------- 비용
def costs(reg, vil, kinds):
    """행정동별 수단별 비용 계산에 필요한 값: 면적 등급, 노선버스 종류, 행복택시 지원금"""
    q = pd.to_numeric(reg["area_km2"], errors="coerce").rank(pct=True)
    size = np.where(q > 2 / 3, 2, np.where(q <= 1 / 3, 0, 1))          # 면적 없으면 가운데
    sub = pd.to_numeric(vil["happy_subsidy_month"], errors="coerce").fillna(0)
    by_dong = sub.groupby(vil["dong_cd"]).sum()
    n_li = (sub > 0).groupby(vil["dong_cd"]).sum()
    op = by_dong[by_dong > 0]
    happy_avg2 = float(op.mean() * HAPPY_MONTHS) if len(op) else np.nan
    info = {}
    for cd, sz in zip(reg["dong_cd"], size):
        info[cd] = {"size": int(sz), "kind": kinds.get(cd, ("CITY_BUS", None)), "has_route_info": cd in kinds,
                    "happy2": float(by_dong.get(cd, 0)) * HAPPY_MONTHS, "happy_li": int(n_li.get(cd, 0))}
    return info, happy_avg2


def cost_of(mode, c, happy_avg2, has_bus):
    """(대수, 비용(원), 투입안, 비용 근거, 차량비 참고). 계산 못 하면 비용 None"""
    if mode == "DRT":
        u = UNITS["DRT"][c["size"]]
        return u, u * COST_DAY["DRT"] * DAYS, f"똑버스 {u}대", \
            f"똑버스 1대 1일 {_man(COST_DAY['DRT'])} × {u}대 × {DAYS}일", None
    if mode == "BUS":
        u = UNITS["BUS"][c["size"]]
        kind, route = c["kind"]
        basis = (f"주 노선 {route or '-'}({BUS_NM[kind]})" if c["has_route_info"]
                 else "노선 정보 없음 → 시내버스 단가")
        return u, u * COST_DAY[kind] * DAYS, f"{BUS_NM[kind]} 경유 노선 증차 ({u}대)", \
            f"{basis} · 1대 1일 {_man(COST_DAY[kind])} × {u}대 × {DAYS}일", (None if has_bus else VEHICLE_NOTE[kind])
    if mode == "HAPPY_TAXI":
        if c["happy2"] > 0:
            return c["happy_li"], round(c["happy2"]), f"행복택시 (리 {c['happy_li']}곳)", \
                f"운영 중인 리 {c['happy_li']}곳 월평균 지원금 합 × {HAPPY_MONTHS}", None
        if np.isnan(happy_avg2):
            return None, None, "행복택시 새로 도입", "운영 중인 읍·면이 없어 비용을 정할 수 없음", None
        return None, round(happy_avg2), "행복택시 새로 도입", f"운영 중인 읍·면 평균 {HAPPY_MONTHS}개월 지원금", None
    return None, None, None, None, None


# ---------------------------------------------------------------- 한 기간
def _rank(df, cols, asc):
    """정렬 순서대로 1, 2, 3 … (같은 값도 정렬 기준이 다 같으면 지역 코드 순)"""
    order = df.sort_values(cols + ["dong_cd"], ascending=asc + [True]).index
    return pd.Series(range(1, len(order) + 1), index=order)


def one_period(p, year, D, cinfo, happy_avg2):
    month = np.nan if p == str(year) else int(p[4:])
    pri = D["pri"]
    pri = pri[(pri["unit"] == "dong") & (pri["month"].isna() if pd.isna(month) else pri["month"] == month)]
    if pri.empty:
        raise JobError("E_NO_DATA", f"ml_priority {p} 없음 (run_all.py 먼저)")
    sc = pri.set_index(pri["region_cd"].astype(str))["score_total"]
    rk = pd.to_numeric(pri.set_index(pri["region_cd"].astype(str))["rank_total"], errors="coerce")
    pk = picks(D["exp"], D["su"], month)
    reg = D["reg"]
    has_bus = dict(zip(reg["dong_cd"], pd.to_numeric(reg["n_stops"], errors="coerce").fillna(0) > 0))

    rows = []
    for r in reg.itertuples():
        cd = r.dong_cd
        score = float(sc.get(cd, np.nan))
        rank = float(rk.get(cd, np.nan))
        mode, cur = pk.get(cd, (None, False))
        u, cost, action, basis, vnote = cost_of(mode, cinfo[cd], happy_avg2, has_bus.get(cd, False))
        if np.isnan(rank) or rank > CANDIDATE_TOP:
            status, note = "후보 아님", (f"종합 취약도 TOP {CANDIDATE_TOP} 밖 ({int(rank)}위, {score:.1f}점)"
                                       if not np.isnan(rank) else "종합 취약도 결과 없음")
        elif mode == "KEEP":
            status, note = "현행 유지", "AI 추천이 현행 유지"
        elif mode == "DRT" and cur:
            status, note = "운영 중", "이미 똑버스가 운영 중이라 새로 드는 비용 없음"
        elif mode is None or cost is None:
            status, note = "후보 아님", "추천 수단 또는 비용을 정할 수 없음"
        else:
            status, note = "후보", None
        if status != "후보":            # 비용·투입안은 후보(포함·예산 밖)만. 수단(AI 추천 1)은 화면용으로 남김
            u = cost = action = basis = vnote = None
        pop65 = pd.to_numeric(pd.Series([r.pop65]), errors="coerce").iloc[0]
        rows.append({"dong_cd": cd, "region_nm": r.dong_nm, "gu": r.gu, "status": status, "note": note,
                     "mode": mode, "mode_nm": "현행 유지" if mode == "KEEP" else config.MODES.get(mode),
                     "action": action, "units": u, "cost": cost, "cost_basis": basis, "vehicle_note": vnote,
                     "score_total": score, "pop65": None if pd.isna(pop65) else int(pop65)})
    df = pd.DataFrame(rows)
    df.index = df["dong_cd"].to_numpy()          # 행 이름 = 행정동 코드 (컬럼 dong_cd 도 그대로 둠)
    cand = df[df["status"] == "후보"].copy()
    cand["effect"] = cand["score_total"] * cand["pop65"].fillna(0)
    cand["eff_per_eok"] = cand["effect"] / (cand["cost"].astype(float) / 1e8)
    cand["rank_eff"] = _rank(cand, ["eff_per_eok", "score_total"], [False, False])
    cand["rank_score"] = _rank(cand, ["score_total", "eff_per_eok"], [False, False])
    cand["bal"] = (cand["rank_eff"] + cand["rank_score"]) / 2
    cand["rank_balance"] = _rank(cand, ["bal", "score_total"], [True, False])
    for c in ["effect", "eff_per_eok", "rank_eff", "rank_score", "rank_balance"]:
        df[c] = cand[c]

    total = float(cand["cost"].sum())
    budget, rule = (BUDGET, "기본 예산") if total >= BUDGET else \
        (round(total / 2), "후보 전체 비용이 기본 예산보다 작아 합계의 절반")

    regions, summary = [], []
    for no, (nm, desc, basis, rk_nm) in PLANS.items():
        key = {1: "rank_eff", 2: "rank_score", 3: "rank_balance"}[no]
        d = df.copy()
        d["plan"], d["plan_nm"], d["plan_rank"] = no, nm, d[key]
        d["is_included"], d["seq"], d["budget_left_at_skip"] = False, pd.NA, pd.NA
        left, seq = float(budget), 0
        for cd in cand.sort_values(key).index:
            c = float(cand.at[cd, "cost"])
            if c <= left:
                seq += 1
                left -= c
                d.loc[cd, ["status", "is_included", "seq"]] = ["포함", True, seq]
            else:
                d.loc[cd, ["status", "budget_left_at_skip"]] = ["예산 밖", round(left)]
        d["line"] = [_line(no, x, rk_nm) for x in d.itertuples()]
        regions.append(d)
        inc = d[d["is_included"]]
        summary.append({"plan": no, "plan_nm": nm, "plan_desc": desc, "sort_basis": basis,
                        "budget": int(budget), "budget_rule": rule, "total_cost": int(inc["cost"].sum()),
                        "budget_left": int(round(left)), "n_included": len(inc), "n_candidates": len(cand),
                        "n_skipped": int((d["status"] == "예산 밖").sum()),
                        "pop65_total": int(inc["pop65"].fillna(0).sum()),
                        "n_bus": int((inc["mode"] == "BUS").sum()), "n_drt": int((inc["mode"] == "DRT").sum()),
                        "n_happy": int((inc["mode"] == "HAPPY_TAXI").sum()), "days": DAYS})
    return pd.concat(regions, ignore_index=True), pd.DataFrame(summary)


def _line(no, x, rk_nm):
    """상세 화면 안별 한 줄"""
    if x.status == "포함":
        return f"{no}안 ✓ {x.mode_nm} · 추정 {_eok(x.cost)} (투입 {x.seq}순위)"
    if x.status == "예산 밖":
        return (f"{no}안 – 예산 범위 밖 ({rk_nm} {int(x.plan_rank)}위) · "
                f"{x.action} {_eok(x.cost)}, 남은 예산 {_eok(x.budget_left_at_skip)}보다 커서 건너뜀")
    return f"{no}안 – {x.status} · {x.note}"


# ---------------------------------------------------------------- 전체
REGION_COLS = ["period", "plan", "plan_nm", "unit", "region_cd", "region_nm", "parent_cd", "gu", "status",
               "is_included", "seq", "plan_rank", "mode", "mode_nm", "action", "units", "cost", "cost_basis",
               "vehicle_note", "score_total", "pop65", "effect", "eff_per_eok", "rank_eff", "rank_score",
               "rank_balance", "budget_left_at_skip", "note", "line"]
INT_COLS = ["seq", "plan_rank", "units", "cost", "pop65", "rank_eff", "rank_score", "rank_balance",
            "budget_left_at_skip"]


def periods_of(year, month, pri):
    """"" 연간 / "02" 그 달 / "all" 연간 + ml_ 결과가 있는 달 전부"""
    have = sorted(int(m) for m in pri.loc[pri["unit"] == "dong", "month"].dropna().unique())
    if month == "":
        return [str(year)]
    if month == "all":
        return [str(year)] + [f"{year}{m:02d}" for m in have]
    if int(month) not in have:
        raise JobError("E_NO_DATA", f"{year}{month} ml_ 결과 없음 (run_all.py 먼저)")
    return [f"{year}{month}"]


def run(year, month):
    """→ ({"plan_region": 표, "plan_summary": 표}, 저장할 기간 목록)"""
    D = load(year)
    periods = periods_of(year, month, D["pri"])
    cinfo, happy_avg2 = costs(D["reg"], D["vil"], D["kinds"])
    vil = D["vil"]
    regs, sums = [], []
    for p in periods:
        d, s = one_period(p, year, D, cinfo, happy_avg2)
        d["unit"], d["region_cd"], d["parent_cd"] = "dong", d["dong_cd"], pd.NA
        # 리: 소속 읍·면(행정동) 결과를 그대로
        li = vil[["li_cd", "li_nm", "dong_cd", "emd"]].merge(d, on="dong_cd", how="inner")
        li["unit"], li["region_cd"], li["parent_cd"] = "village", li["li_cd"], li["dong_cd"]
        li["region_nm"] = li["emd"] + " " + li["li_nm"]
        regs += [d.assign(period=p), li.assign(period=p)]
        sums.append(s.assign(period=p))
    region = pd.concat(regs, ignore_index=True)[REGION_COLS]
    for c in INT_COLS:
        region[c] = pd.to_numeric(region[c], errors="coerce").round().astype("Int64")
    region["is_included"] = region["is_included"].astype(bool)
    region = region.sort_values(["period", "plan", "unit", "seq", "plan_rank", "region_cd"],
                                key=lambda s: s.map({"dong": 0, "village": 1}) if s.name == "unit" else s,
                                na_position="last").reset_index(drop=True)
    summary = pd.concat(sums, ignore_index=True)
    summary = summary[["period"] + [c for c in summary.columns if c != "period"]]
    return {"plan_region": region, "plan_summary": summary}, periods

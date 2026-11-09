"""
용인 교통 사각지대 서비스 – 머신러닝·점수 계산 (적합성 평가, 우선순위 평가, 지역 유형, AI 추천 카드)

    python run_all.py                     ← 보통은 아래 YEAR·MONTH 만 바꾸고 이것만 실행 (VS Code ▷ 버튼도 됨)
    python run_all.py --year 2026 --month 03   # 위 값 대신 이번만 다르게
    python run_all.py --no-db-write       # 계산만 하고 DB에는 안 씀 (outputs/ CSV로 확인)
    ※ 로그 파일·화면 출력 없음. 종료 코드: 0 성공 / 1 실패 / 2 이미 다른 실행 중(바로 끝남) / 130 사용자가 멈춤
    ※ 한 번에 하나만 실행 (DB 잠금 GET_LOCK('yong_ml'). 연도·달이 달라도 동시에는 안 돎)
    python run_all.py --no-llm            # LLM 안 부름 (저장된 LLM 카드는 그대로 쓰고, 없으면 템플릿 문장)
    python run_all.py --llm-scope mvp     # LLM 은 백암·원삼만 (나머지 지역은 템플릿 문장)
    python run_all.py --fill-template     # 카드(ml_explanation)만 저장: 템플릿이거나 아직 없는 카드만 LLM 으로 채움
    python run_all.py --regen-llm         # 저장된 LLM 카드도 전부 새로 생성 (프롬프트를 바꿨을 때)

AI 카드(LLM)는 이미 저장된 카드를 다시 만들지 않는다:
  같은 연·월 카드가 LLM 으로 저장돼 있고 넘기는 사실·추천 결과가 그대로면 그대로 씀 → LLM 호출 없음.
  데이터가 바뀐 지역, 템플릿이던 지역, 처음 만드는 달만 LLM 을 부름 (한 달 110곳 ≈ 130회, 무료 하루 500회)

저장: MONTH 로 정한 기간 행만 저장하고, 미리 저장해 둔 다른 기간은 그대로
  MONTH = ""     → 2025 연간만 (추가, 이미 있으면 덮어쓰기)
  MONTH = "02"   → 2025년 2월만 (추가, 이미 있으면 덮어쓰기)
  MONTH = "all"  → 2025 연간 + 1~12월 전부 (LLM 호출 많음: 약 1,700회)

만드는 테이블 (yongin DB) — 모두 year + month (연간 행은 month 빈 값, 월별 행은 month 1~12)
  ml_priority      우선순위 평가: 항목별 점수(0~100) + 종합 취약도·순위·취약지역 여부
  ml_suitability   적합성 평가: 지역(리·행정동) × 노선버스·똑버스·행복택시 점수
  ml_region        지역 유형(6개) + 유형 칩 + 지표 신뢰도 낮음 한 줄
  ml_explanation   AI 추천 해결방안 카드(현행 유지 지역 4개, 나머지 3개) + 대시보드 한 줄
  ml_model_info    모델 정확도 (모델은 연간 데이터로 학습. 실행한 기간 행마다 그때 모델 정확도)
"""
# --- 이 폴더의 가상환경(.venv)이 있으면 그걸로 실행 (VS Code 실행 버튼용) ---
import os as _os, sys as _sys
from pathlib import Path as _P
_VPY = _P(__file__).resolve().parent / ".venv" / "bin" / "python"
if _VPY.exists() and _P(_sys.prefix).resolve() != _VPY.parents[1].resolve() and not _os.environ.get("YONG_NO_VENV"):
    _os.execv(str(_VPY), [str(_VPY), *_sys.argv])
# -----------------------------------------------------------------------------

# ============================== 여기만 바꿔서 실행 ==============================
YEAR = "2025"     # 기준연도. yongin DB feat_ 표에서 이 연도(year)만 골라 씀 → 결과도 이 연도로 추가
MONTH = "9"        # ""    = 연간만 저장
                  # "03"  = 3월만 저장
                  # "all" = 연간 + 1~12월 전부 저장 (LLM 카드 약 1,700회 → 무료 한도로 며칠)
                  # 저장한 기간만 바뀌고, 미리 저장해 둔 다른 기간(연간·다른 달)은 그대로
                  # (월별 점수는 어느 달을 저장하든 그 해 12개월을 같이 놓고 계산 → 값은 같음)
# ==============================================================================

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from yongml import explain, facts, features, io_db, priority, region_type, suitability  # noqa: E402
from yongml.errors import JobError  # noqa: E402


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-db-write", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--llm-scope", choices=["all", "mvp"], default="all")
    ap.add_argument("--fill-template", action="store_true", help="카드만 저장: 템플릿이거나 없는 카드만 LLM 으로")
    ap.add_argument("--regen-llm", action="store_true", help="저장된 LLM 카드도 새로 생성")
    ap.add_argument("--year", default=YEAR)
    ap.add_argument("--month", default=MONTH)
    a = ap.parse_args()
    config.YEAR = str(a.year)
    m = str(a.month).strip().lower()
    month = "all" if m == "all" else (f"{int(m):02d}" if m else "")
    trained_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    if a.fill_template and a.no_llm:
        raise JobError("E_ARG", "--fill-template 는 LLM 을 써야 해서 --no-llm 과 같이 못 씀")

    T = io_db.load(config.YEAR)
    vil, reg, stp = features.build(T)
    vil_m, reg_m = features.monthly(vil, reg, T)
    su, summ, infos, vil, reg, vil_m, reg_m, stp = suitability.run(vil, reg, stp, vil_m, reg_m)
    if month not in ("", "all") and f"{config.YEAR}{month}" not in set(su["period"]):
        raise JobError("E_NO_DATA", f"{config.YEAR}{month} 월별 없음")
    pri = priority.run(vil, reg, vil_m, reg_m, summ)

    # 저장할 기간: "" 연간 / "02" 그 달 / "all" 연간 + 1~12월. 지역 유형·AI 카드는 이 기간만 만듦
    months = sorted(p for p in su["period"].astype(str).unique() if len(p) == 6)
    periods = ([config.YEAR] if month == "" else [f"{config.YEAR}{month}"] if month != "all"
               else [config.YEAR] + months)
    saved = io_db.read_saved("ml_explanation", config.YEAR)       # 이미 저장된 카드 (다시 안 만들려고)
    if saved.empty:
        saved = pd.DataFrame(columns=["unit", "region_cd", "month"])
    explain.STATE["quota"] = False
    rgns, exps = [], []
    for p in periods:
        v, r, city_m, rm, label = _period_frames(p, vil, reg, vil_m, reg_m, T)
        R = facts.refs(v, r, T["feat_city"], city_m)          # 비교 기준 평균 (월별은 그 달)
        rgn, rel_cautions, goods = region_type.run(v, r, R, rm, label)
        old = saved[saved["month"].isna()] if p == config.YEAR else saved[saved["month"] == int(p[4:])]
        reuse = {(u, c): row for u, c, row in zip(old["unit"], old["region_cd"], old.to_dict("records"))}
        exp = explain.run(v, r, su, R, rgn, rel_cautions, T["complaint"], stp=stp, use_llm=not a.no_llm,
                          scope=a.llm_scope, goods=goods, pri=pri, period=p, label=label, reuse=reuse,
                          regen=a.regen_llm)
        rgns.append(rgn.assign(period=p))
        exps.append(exp.assign(period=p))
    exp = pd.concat(exps, ignore_index=True)

    if a.fill_template:
        tables = {"ml_explanation": exp}
    else:
        tables = {
            "ml_priority": pri,
            "ml_suitability": suitability.table(su),
            "ml_region": pd.concat(rgns, ignore_index=True),
            "ml_explanation": exp,
            "ml_model_info": pd.concat([suitability.model_info_table(infos, trained_at).assign(period=p)
                                        for p in periods], ignore_index=True),
        }
    io_db.save(tables, config.YEAR, periods, to_db=not a.no_db_write)


def _period_frames(p, vil, reg, vil_m, reg_m, T):
    """한 기간의 지역 값·시 평균·택시 원본
    월별: 그 달 행. 카드의 택시 근거 지표는 그래프와 같은 숫자가 나오게 그 달 원본 값(feat_region_month)으로
    (점수 계산용 택시 값은 최근 3개월이라 화면 그래프와 다름)"""
    city_m, rm = T["feat_city_month"], T["feat_region_month"]
    if p == config.YEAR:
        return vil, reg, city_m, rm, "연간"
    v = vil_m[vil_m["period"] == p].reset_index(drop=True)
    r = reg_m[reg_m["period"] == p].reset_index(drop=True)
    rm = rm[rm["ym"] == p]
    city_m = city_m[city_m["ym"] == p]
    raw = rm.set_index("dong_cd")
    req = pd.to_numeric(raw["taxi_req"], errors="coerce")
    match = pd.to_numeric(raw["taxi_match_rate"], errors="coerce")
    per1000 = req / r.set_index("dong_cd")["pop"].reindex(req.index) * 1000
    for df in (v, r):
        df["taxi_match"] = df["dong_cd"].map(match)
        df["taxi_per1000"] = df["dong_cd"].map(per1000).round(2)
    return v, r, city_m, rm, f"{int(p[:4])}년 {int(p[4:])}월"


def main():
    """종료 코드: 0 성공 / 1 실패 / 2 이미 다른 실행이 돌고 있음 / 130 사용자가 멈춤. 로그·화면 출력은 없음"""
    try:
        with io_db.run_lock():
            run()
    except JobError as e:
        sys.exit(2 if e.code == "E_BUSY" else 1)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001
        sys.exit(1)


if __name__ == "__main__":
    main()

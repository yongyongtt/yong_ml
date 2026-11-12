"""
용인 교통 사각지대 서비스 – 통합 최적안(A안): 어느 지역부터 무엇을 넣을지 1·2·3안 (효율형·형평형·균형형)

    python run_plan.py                     ← 보통은 아래 YEAR·MONTH 만 바꾸고 이것만 실행 (VS Code ▷ 버튼도 됨)
    python run_plan.py --year 2026 --month 03   # 위 값 대신 이번만 다르게
    python run_plan.py --no-db-write       # 계산만 하고 DB 에는 안 씀 (outputs/<연도>/plan_*.csv 로 확인)
    ※ 머신러닝·LLM 없음. run_all.py 가 저장한 같은 연·월 ml_ 결과를 읽어서 계산만 함 (몇 초)
      → run_all.py 를 먼저 돌려 두어야 함 (그 연·월 결과가 없으면 종료 코드 1)
    ※ 로그 파일·화면 출력 없음. 종료 코드: 0 성공 / 1 실패 / 2 이미 다른 실행 중(바로 끝남) / 130 사용자가 멈춤
    ※ run_all.py 와 같은 잠금 GET_LOCK('yong_ml') → run_all.py 가 저장하는 도중의 결과를 읽지 않음
단가·예산·대수 기준과 계산 방법: yongml/plan.py 맨 위

저장: MONTH 로 정한 기간 행만 저장하고, 미리 저장해 둔 다른 기간은 그대로 (run_all.py 와 같음)
  MONTH = ""     → 2025 연간만
  MONTH = "02"   → 2025년 2월만
  MONTH = "all"  → 2025 연간 + 1~12월 (ml_ 결과가 있는 달만)

만드는 테이블 (yongin DB) — 모두 year + month (연간 행은 month 빈 값)
  plan_region   안(1·2·3) × 지역(행정동 + 리): 상태(포함·예산 밖·후보 아님·현행 유지·운영 중), 투입 순서,
                수단·대수·추정 비용(원, 2개월), 기준 순위, 상세 화면 한 줄(line)
                리(unit=village)는 소속 읍·면(parent_cd) 결과를 그대로
  plan_summary  안별 요약 (메인 카드): 예산, 총비용, 포함 지역 수, 65세 이상 인구, 수단 구성
"""
# --- 이 폴더의 가상환경(.venv)이 있으면 그걸로 실행 (VS Code 실행 버튼용) ---
import os as _os, sys as _sys
from pathlib import Path as _P
_VPY = _P(__file__).resolve().parent / ".venv" / "bin" / "python"
if _VPY.exists() and _P(_sys.prefix).resolve() != _VPY.parents[1].resolve() and not _os.environ.get("YONG_NO_VENV"):
    _os.execv(str(_VPY), [str(_VPY), *_sys.argv])
# -----------------------------------------------------------------------------

# ============================== 여기만 바꿔서 실행 ==============================
YEAR = "2025"     # 기준연도. 이 연도의 ml_ 결과·feat_ 표를 읽음
MONTH = "all"        # ""    = 연간만 저장
                  # "03"  = 3월만 저장
                  # "all" = 연간 + 1~12월 전부 저장
                  # 저장한 기간만 바뀌고, 미리 저장해 둔 다른 기간은 그대로
# ==============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from yongml import io_db, plan  # noqa: E402
from yongml.errors import JobError  # noqa: E402


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-db-write", action="store_true")
    ap.add_argument("--year", default=YEAR)
    ap.add_argument("--month", default=MONTH)
    a = ap.parse_args()
    config.YEAR = str(a.year)
    m = str(a.month).strip().lower()
    month = "all" if m == "all" else (f"{int(m):02d}" if m else "")
    tables, periods = plan.run(config.YEAR, month)
    io_db.save(tables, config.YEAR, periods, to_db=not a.no_db_write)


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

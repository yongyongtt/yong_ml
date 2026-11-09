"""
yong_ml 실행 API — run_all.py 를 year·month 로 실행하는 FastAPI

    uvicorn api:app --host 127.0.0.1 --port 8000

    POST /api/ml/run          {"year": 2025, "month": 1}  ← python run_all.py --year 2025 --month 01 을 뒤에서 시작
                              {"year": 2025}              ← month 를 빼면 연간 (--month "")
    GET  /api/ml/run/status   지금 실행 중이거나 마지막에 끝난 실행 하나

한 기간 실행은 약 15분이라 POST 는 끝날 때까지 기다리지 않고 바로 응답하고, 끝났는지는 status 로 조회
  POST  202 RUNNING   접수 (뒤에서 실행 시작)
        409 RUNNING   이미 다른 실행이 돌고 있음 (API 로 시작한 것, 서버에서 run_all.py 를 직접 돌린 것 모두)
        422           year·month 형식이 틀림
  GET   200           status: NONE / RUNNING / SUCCEEDED / FAILED / BUSY
run_all.py 는 그대로 씀 (한 번에 하나만 도는 잠금 GET_LOCK('yong_ml') 도 run_all.py 안에 있음)
실행 기록은 이 API 서버 메모리에만 있음. API 서버를 멈추거나 다시 켜도 이미 시작한 실행은 끝까지 가고 DB 에 저장됨
"""
import datetime as dt
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
EXPECTED_SEC = 15 * 60                  # 한 기간(연간 또는 한 달) 예상 실행 시간
DONE = {0: "SUCCEEDED", 2: "BUSY"}      # run_all.py 종료 코드 → status (그 밖의 코드 = FAILED)

app = FastAPI(title="yong_ml 실행 API", version="2.0.0",
              description="year·month 로 run_all.py 를 실행해 yongin DB 의 ml_ 테이블을 채움 (바로 응답, 상태는 따로 조회)")


class RunRequest(BaseModel):
    year: int = Field(..., ge=2000, le=2100, description="기준 연도", examples=[2025])
    month: Optional[int] = Field(None, ge=1, le=12, description="1~12. 빼거나 null 이면 연간", examples=[1])


class RunStatus(BaseModel):
    year: Optional[int] = Field(None, description="연도. 서버에서 직접 돌린 실행이면 null")
    month: Optional[int] = Field(None, description="월. 연간이면 null")
    status: str = Field(..., description="NONE | RUNNING | SUCCEEDED | FAILED | BUSY")
    exitCode: Optional[int] = Field(None, description="run_all.py 종료 코드. 실행 중이면 null")
    startedAt: Optional[str] = Field(None, description="시작 시각 (ISO 8601)")
    finishedAt: Optional[str] = Field(None, description="끝난 시각. 실행 중이면 null")
    elapsedSec: Optional[int] = Field(None, description="걸린 시간(초). 실행 중이면 지금까지")
    expectedSec: int = Field(EXPECTED_SEC, description="예상 실행 시간(초)")


_guard = threading.Lock()   # POST 두 개가 동시에 들어와도 하나만 시작
_last: dict = {}            # 지금 실행 중이거나 마지막 실행 (API 로 시작한 것만)


def _now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _json(code, body):
    return JSONResponse(status_code=code, content=jsonable_encoder(body))


def _body(run):
    if not run:
        return RunStatus(status="NONE")
    done = run["done"]
    return RunStatus(year=run["year"], month=run["month"],
                     status=DONE.get(run["exitCode"], "FAILED") if done else "RUNNING",
                     exitCode=run["exitCode"], startedAt=run["startedAt"], finishedAt=run["finishedAt"],
                     elapsedSec=run["elapsed"] if done else int(time.monotonic() - run["t0"]))


def _watch(run):
    """실행이 끝나면 종료 코드·끝난 시각 기록 (done 은 맨 마지막에 바꿈)"""
    code = run["proc"].wait()
    run.update(exitCode=code, finishedAt=_now(), elapsed=int(time.monotonic() - run["t0"]))
    run["done"] = True


def _outside_running():
    """API 밖(서버에서 run_all.py 를 직접 실행, API 재시작 전에 시작한 실행)에서 돌고 있는지: MariaDB IS_USED_LOCK.
    확인이 안 되면(DB 연결 실패 등) False → 일단 시작하고, 겹치면 run_all.py 가 종료 코드 2(BUSY)로 끝남"""
    try:
        from sqlalchemy import text
        from yongml import io_db
        eng = io_db.engine()
        try:
            with eng.connect() as con:
                return con.execute(text("SELECT IS_USED_LOCK(:n)"), {"n": io_db.LOCK_NAME}).scalar() is not None
        finally:
            eng.dispose()
    except Exception:  # noqa: BLE001
        return False


@app.post("/api/ml/run", status_code=202, response_model=RunStatus, summary="run_all.py 실행 시작 (바로 응답)",
          responses={409: {"model": RunStatus, "description": "이미 다른 실행이 돌고 있음"},
                     500: {"model": RunStatus, "description": "실행을 시작하지 못함"}})
def run(req: RunRequest):
    global _last
    with _guard:
        if _last and not _last["done"]:
            return _json(409, _body(_last))
        if _outside_running():
            return _json(409, RunStatus(status="RUNNING"))
        month = "" if req.month is None else f"{req.month:02d}"
        new = {"year": req.year, "month": req.month, "t0": time.monotonic(), "startedAt": _now(),
               "finishedAt": None, "exitCode": None, "elapsed": None, "done": False}
        try:
            new["proc"] = subprocess.Popen(
                [sys.executable, str(ROOT / "run_all.py"), "--year", str(req.year), "--month", month],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,   # API 서버를 멈춰도 실행은 끝까지 (저장 도중에 같이 끊기지 않게)
            )
        except OSError:
            new.update(finishedAt=_now(), elapsed=0, done=True)
            _last = new
            return _json(500, _body(new))
        _last = new
        threading.Thread(target=_watch, args=(new,), daemon=True).start()
        return _json(202, _body(new))


@app.get("/api/ml/run/status", response_model=RunStatus, summary="지금 실행 중이거나 마지막에 끝난 실행")
def status():
    run = _last
    if run and not run["done"]:
        return _body(run)
    if _outside_running():
        return RunStatus(status="RUNNING")   # API 밖에서 도는 실행: 연·월·시각은 모름
    return _body(run)

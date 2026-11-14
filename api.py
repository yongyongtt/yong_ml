"""
yong_ml 실행 API — run_all.py(ML)·run_plan.py(통합 최적안)를 year·month 로 실행하는 FastAPI

    uvicorn api:app --host 127.0.0.1 --port 8000

    POST /api/ml/run            {"year": 2025, "month": 1}  ← python run_all.py --year 2025 --month 01 을 뒤에서 시작
                                {"year": 2025}              ← month 를 빼면 연간 (--month "")
    GET  /api/ml/run/status     지금 실행 중이거나 마지막에 끝난 ML 실행 하나
    POST /api/plan/run          {"year": 2025, "month": 1}  ← python run_plan.py --year 2025 --month 01 (파라미터는 ML 과 같음)
    GET  /api/plan/run/status   지금 실행 중이거나 마지막에 끝난 통합 최적안 실행 하나

ML 한 기간 실행은 약 15분이라 POST 는 끝날 때까지 기다리지 않고 바로 응답하고, 끝났는지는 status 로 조회
통합 최적안은 ML 결과(ml_ 표)를 읽어서 계산만 해서 몇 초. 같은 연·월 ML 결과가 먼저 있어야 함 (없으면 FAILED)
  POST  202 RUNNING   접수 (뒤에서 실행 시작)
        409 RUNNING   이미 다른 실행이 돌고 있음 (ML·통합 최적안 중 아무거나. 서버에서 직접 돌린 것 포함)
        422           year·month 형식이 틀림
  GET   200           status: NONE / RUNNING / SUCCEEDED / FAILED / BUSY
두 스크립트는 같은 잠금 GET_LOCK('yong_ml') 을 써서 한 번에 하나만 돎 → API 도 둘 중 하나가 돌면 다른 하나는 409
run_all.py·run_plan.py 는 그대로 씀. 실행 기록은 이 API 서버 메모리에만 있음
API 서버를 멈추거나 다시 켜도 이미 시작한 실행은 끝까지 가고 DB 에 저장됨 (Docker 컨테이너를 멈추면 같이 끊김)
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
JOBS = {   # job: (실행할 스크립트, 예상 실행 시간(초))
    "ml": ("run_all.py", 15 * 60),       # 한 기간(연간 또는 한 달)
    "plan": ("run_plan.py", 60),         # 보통 몇 초
}
EXPECTED_SEC = JOBS["ml"][1]
DONE = {0: "SUCCEEDED", 2: "BUSY"}      # 스크립트 종료 코드 → status (그 밖의 코드 = FAILED)

app = FastAPI(title="yong_ml 실행 API", version="2.1.0",
              description="year·month 로 run_all.py(ML)·run_plan.py(통합 최적안)를 실행해 yongin DB 를 채움 "
                          "(바로 응답, 상태는 따로 조회)")


class RunRequest(BaseModel):
    year: int = Field(..., ge=2000, le=2100, description="기준 연도", examples=[2025])
    month: Optional[int] = Field(None, ge=1, le=12, description="1~12. 빼거나 null 이면 연간", examples=[1])


class RunStatus(BaseModel):
    job: Optional[str] = Field(None, description="ml | plan. 서버에서 직접 돌린 실행이면 null")
    year: Optional[int] = Field(None, description="연도. 서버에서 직접 돌린 실행이면 null")
    month: Optional[int] = Field(None, description="월. 연간이면 null")
    status: str = Field(..., description="NONE | RUNNING | SUCCEEDED | FAILED | BUSY")
    exitCode: Optional[int] = Field(None, description="스크립트 종료 코드. 실행 중이면 null")
    startedAt: Optional[str] = Field(None, description="시작 시각 (ISO 8601)")
    finishedAt: Optional[str] = Field(None, description="끝난 시각. 실행 중이면 null")
    elapsedSec: Optional[int] = Field(None, description="걸린 시간(초). 실행 중이면 지금까지")
    expectedSec: int = Field(EXPECTED_SEC, description="예상 실행 시간(초)")


_guard = threading.Lock()                 # POST 두 개가 동시에 들어와도 하나만 시작
_runs: dict = {job: {} for job in JOBS}   # job 별 지금 실행 중이거나 마지막 실행 (API 로 시작한 것만)


def _now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _json(code, body):
    return JSONResponse(status_code=code, content=jsonable_encoder(body))


def _running(job):
    run = _runs[job]
    return bool(run) and not run["done"]


def _body(job, run):
    if not run:
        return RunStatus(job=job, status="NONE", expectedSec=JOBS[job][1])
    done = run["done"]
    return RunStatus(job=job, year=run["year"], month=run["month"],
                     status=DONE.get(run["exitCode"], "FAILED") if done else "RUNNING",
                     exitCode=run["exitCode"], startedAt=run["startedAt"], finishedAt=run["finishedAt"],
                     elapsedSec=run["elapsed"] if done else int(time.monotonic() - run["t0"]),
                     expectedSec=JOBS[job][1])


def _watch(run):
    """실행이 끝나면 종료 코드·끝난 시각 기록 (done 은 맨 마지막에 바꿈)"""
    code = run["proc"].wait()
    run.update(exitCode=code, finishedAt=_now(), elapsed=int(time.monotonic() - run["t0"]))
    run["done"] = True


def _lock_held():
    """DB 잠금이 잡혀 있는지 (API 밖에서 직접 돌린 실행, API 재시작 전에 시작한 실행 포함): MariaDB IS_USED_LOCK.
    확인이 안 되면(DB 연결 실패 등) False → 일단 시작하고, 겹치면 스크립트가 종료 코드 2(BUSY)로 끝남"""
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


def _start(job, req: RunRequest):
    with _guard:
        for j in JOBS:                                   # API 로 시작한 실행이 돌고 있으면 (ML·최적안 모두)
            if _running(j):
                return _json(409, _body(j, _runs[j]))
        if _lock_held():                                 # API 밖에서 도는 실행: 연·월·종류는 모름
            return _json(409, RunStatus(status="RUNNING", expectedSec=JOBS[job][1]))
        script = JOBS[job][0]
        month = "" if req.month is None else f"{req.month:02d}"
        new = {"year": req.year, "month": req.month, "t0": time.monotonic(), "startedAt": _now(),
               "finishedAt": None, "exitCode": None, "elapsed": None, "done": False}
        try:
            new["proc"] = subprocess.Popen(
                [sys.executable, str(ROOT / script), "--year", str(req.year), "--month", month],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,   # API 서버를 멈춰도 실행은 끝까지 (저장 도중에 같이 끊기지 않게)
            )
        except OSError:
            new.update(finishedAt=_now(), elapsed=0, done=True)
            _runs[job] = new
            return _json(500, _body(job, new))
        _runs[job] = new
        threading.Thread(target=_watch, args=(new,), daemon=True).start()
        return _json(202, _body(job, new))


def _status(job):
    run = _runs[job]
    if _running(job):
        return _body(job, run)
    if not any(_running(j) for j in JOBS) and _lock_held():
        return RunStatus(status="RUNNING", expectedSec=JOBS[job][1])   # API 밖에서 도는 실행: 연·월·종류는 모름
    return _body(job, run)


RESPONSES = {409: {"model": RunStatus, "description": "이미 다른 실행이 돌고 있음 (ML·통합 최적안 중 아무거나)"},
             500: {"model": RunStatus, "description": "실행을 시작하지 못함"}}


@app.post("/api/ml/run", status_code=202, response_model=RunStatus, responses=RESPONSES,
          summary="run_all.py(ML) 실행 시작 (바로 응답)")
def run(req: RunRequest):
    return _start("ml", req)


@app.get("/api/ml/run/status", response_model=RunStatus, summary="지금 실행 중이거나 마지막에 끝난 ML 실행")
def status():
    return _status("ml")


@app.post("/api/plan/run", status_code=202, response_model=RunStatus, responses=RESPONSES,
          summary="run_plan.py(통합 최적안) 실행 시작 (바로 응답, 보통 몇 초)")
def plan_run(req: RunRequest):
    return _start("plan", req)


@app.get("/api/plan/run/status", response_model=RunStatus, summary="지금 실행 중이거나 마지막에 끝난 통합 최적안 실행")
def plan_status():
    return _status("plan")

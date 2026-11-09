"""입력 읽기(yongin DB, 연도로 골라서)와 결과 저장(DB ml_ 테이블에 추가로 쌓기 + outputs/ 에 확인용 CSV)
DB 에 연결이 안 되거나 필요한 표가 없으면 오류 코드(JobError)로 멈춘다 (다른 데이터로 대신 돌리지 않음).
  E_DB_CONFIG  DB 설정 없음          E_DB_DRIVER  DB 패키지 없음       E_DB_CONNECT  DB 연결 실패(번호)
  E_TABLE      표 읽기 실패(번호)     E_NO_DATA    그 연도·달 데이터 없음  E_DB_WRITE    저장 실패(번호)
  E_ARG        실행 옵션을 같이 쓸 수 없음
  E_BUSY       다른 실행이 이미 돌고 있음 (종료 코드 2, 아무것도 안 하고 바로 끝남)
  번호: 1045 계정·비밀번호 틀림, 2003 서버 꺼짐·주소 틀림, 1049 DB 없음, 1146 표 없음

연·월 컬럼 (모든 표 같은 이름)
  - 연간 데이터만 있는 표: year            (예: year 2025)
  - 월별 데이터가 있는 표: year + month    (예: year 2025, month 5 / 연간 행은 month 빈 값)
"""
import os
from contextlib import contextmanager

import pandas as pd

import config
from .errors import JobError, db_errno

ANNUAL_TABLES = ["feat_village", "feat_region", "feat_stop", "feat_region_hour", "feat_city"]   # year 로 고름
MONTHLY_TABLES = ["feat_village_month", "feat_region_month", "feat_city_month"]               # year + month
FEAT_TABLES = ANNUAL_TABLES + MONTHLY_TABLES
CODE_COLS = ["dong_cd", "li_cd", "stop_id", "access_stop_id", "sgis_cd"]
BOOL_COLS = ["is_eupmyeon", "happy_taxi", "small_li", "has_boundary", "runs_le5", "hub_direct", "is_hub",
             "hub_rail", "hub_terminal", "hub_local", "fp_anomaly"]
LONG_TEXT = ("context", "metrics", "options", "reasons", "cautions", "type_chips")


def engine(database=None):
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL
    if os.getenv("DB_URL"):  # 테스트용 (예: sqlite:///test.db)
        return create_engine(os.getenv("DB_URL"), hide_parameters=True)
    url = URL.create("mysql+pymysql", username=config.DB["user"], password=config.DB["password"],
                     host=config.DB["host"], port=config.DB["port"],
                     database=database or config.DB["name"], query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, hide_parameters=True)


def connect():
    """DB 연결. 안 되면 JobError"""
    if not os.getenv("DB_URL") and not config.DB["user"]:
        raise JobError("E_DB_CONFIG", "DB 설정 없음")
    try:
        eng = engine()
        with eng.connect() as con:
            con.exec_driver_sql("SELECT 1")
        return eng
    except ModuleNotFoundError:
        raise JobError("E_DB_DRIVER", "DB 패키지 없음") from None
    except Exception as e:  # noqa: BLE001
        raise JobError("E_DB_CONNECT", f"DB 연결 실패 ({db_errno(e)})") from e



LOCK_NAME = "yong_ml"   # 연도·달 상관없이 한 번에 하나만 실행 (LLM 분당 한도도 같이 지킴)


@contextmanager
def run_lock():
    """run_all.py 가 같은 DB 에서 동시에 두 번 돌지 않게 MariaDB GET_LOCK 으로 잠금.
    이미 다른 실행이 잡고 있으면 기다리지 않고 바로 JobError(E_BUSY).
    잠금은 이 연결에 묶여 있어서 끝나면 풀고, 프로세스가 죽어도 연결이 끊기면서 저절로 풀림"""
    if os.getenv("DB_URL"):      # 테스트용 DB(sqlite)는 잠금 없음
        yield
        return
    from sqlalchemy import text
    eng = connect()
    con = eng.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        # 오래 걸리는 실행(연간 + 12개월은 몇 시간) 중에 DB 가 이 연결을 끊어 잠금이 풀리지 않게
        con.execute(text("SET SESSION wait_timeout = 86400"))
        got = con.execute(text("SELECT GET_LOCK(:n, 0)"), {"n": LOCK_NAME}).scalar()
    except Exception as e:  # noqa: BLE001
        con.close()
        eng.dispose()
        raise JobError("E_DB_CONNECT", f"잠금 실패 ({db_errno(e)})") from e
    if got != 1:
        con.close()
        eng.dispose()
        raise JobError("E_BUSY", "다른 실행이 돌고 있음")
    try:
        yield
    finally:
        try:
            con.execute(text("SELECT RELEASE_LOCK(:n)"), {"n": LOCK_NAME})
        except Exception:  # noqa: BLE001  (연결이 이미 끊겼으면 잠금도 이미 풀림)
            pass
        con.close()
        eng.dispose()

def _read(eng, table):
    try:
        return pd.read_sql(f"SELECT * FROM {table}", eng)
    except Exception as e:  # noqa: BLE001
        raise JobError("E_TABLE", f"{table} 읽기 실패 ({db_errno(e)})") from e


def _tidy(df):
    for c in CODE_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").str.replace(r"\.0$", "", regex=True)
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: str(v).strip().lower() in ("1", "true", "t", "y")).astype(bool)
    return df


def _pick_year(name, df, year):
    """그 연도 행만. 월별 표는 계산용으로 ym(YYYYMM 문자열) 컬럼을 하나 만들어 둠 (DB 에는 안 씀)"""
    if "year" not in df.columns:
        raise JobError("E_TABLE", f"{name} 예전 형식 (year 없음) - yong 전처리를 다시 돌려 주세요")
    out = df[pd.to_numeric(df["year"], errors="coerce") == int(year)].copy()
    if out.empty:
        raise JobError("E_NO_DATA", f"{name} {year}년 없음")
    if name in MONTHLY_TABLES:
        m = pd.to_numeric(out["month"], errors="coerce").astype(int)
        out["ym"] = str(int(year)) + m.astype(str).str.zfill(2)
    return out


def load(year):
    """yongin DB 에서 feat_ 표(그 연도만)와 민원 표를 읽음. 연결이 안 되거나 표·연도가 없으면 멈춤"""
    eng = connect()
    T = {t: _pick_year(t, _read(eng, t), year) for t in FEAT_TABLES}
    T["complaint"] = _read(eng, config.COMPLAINT_TABLE)
    return {k: _tidy(v) for k, v in T.items()}


def read_saved(name, year):
    """이미 저장된 ml_ 표에서 그 연도 행 (표가 없으면 빈 표). year 컬럼은 빼고 돌려줌"""
    from sqlalchemy import inspect
    eng = connect()
    if not inspect(eng).has_table(name):
        return pd.DataFrame()
    df = _read(eng, name)
    if "year" not in df.columns:
        return pd.DataFrame()
    df = df[pd.to_numeric(df["year"], errors="coerce") == int(year)].drop(columns=["year"])
    if "month" not in df.columns:      # 예전 형식(연간만) 표
        df["month"] = pd.NA
    if "region_cd" in df.columns:
        df["region_cd"] = df["region_cd"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df.reset_index(drop=True)


def upsert(eng, name, df, key_cols, dtype=None):
    """추가로 쌓기: 이번에 넣는 (year) 또는 (year, month) 와 같은 행만 지우고 새로 추가. month 빈 값(연간 행)은 그것끼리.
    다른 연도·달은 그대로. 새 컬럼은 표에 추가. 예전 형식(키 컬럼 없음) 표는 새로 만듦"""
    from sqlalchemy import inspect, text
    insp = inspect(eng)
    if not insp.has_table(name):
        df.to_sql(name, eng, if_exists="replace", index=False, chunksize=2000, dtype=dtype)
        return
    cols = [c["name"] for c in insp.get_columns(name)]
    if any(k not in cols for k in key_cols):
        df.to_sql(name, eng, if_exists="replace", index=False, chunksize=2000, dtype=dtype)
        return
    q = eng.dialect.identifier_preparer.quote   # options·rank 같은 이름도 안전하게
    with eng.begin() as con:
        for c in [c for c in df.columns if c not in cols]:
            kind = df[c].dtype.kind
            sqltype = {"i": "BIGINT", "u": "BIGINT", "f": "DOUBLE", "b": "TINYINT"}.get(kind, "TEXT")
            con.execute(text(f"ALTER TABLE {q(name)} ADD COLUMN {q(c)} {sqltype}"))
        for tup in df[key_cols].drop_duplicates().itertuples(index=False):
            conds, params = [], {}
            for i, (k, v) in enumerate(zip(key_cols, tup)):
                if pd.isna(v):
                    conds.append(f"{q(k)} IS NULL")
                else:
                    conds.append(f"{q(k)} = :k{i}")
                    params[f"k{i}"] = int(v)
            con.execute(text(f"DELETE FROM {q(name)} WHERE {' AND '.join(conds)}"), params)
    df.to_sql(name, eng, if_exists="append", index=False, chunksize=2000)


def _year_month(df, year):
    """계산 중 쓰던 period('2025' 연간 / '202503' 월별) → year, month 컬럼 (맨 앞). period 가 없으면 year 만"""
    if "period" in df.columns:
        p = df["period"].astype(str)
        month = pd.array([int(x[4:6]) if len(x) == 6 else None for x in p], dtype="Int64")
        df = df.drop(columns=["period"])
        df.insert(0, "month", month)
    df.insert(0, "year", int(year))
    return df


def save(tables: dict, year, periods, to_db=True):
    """이번에 돌린 기간(periods: "2025" 연간 / "202502" 월별) 행만 저장. CSV 는 outputs/<연도>/ 에, DB 는 추가로 쌓음
    - 모든 ml_ 표: (year, month) 단위로 이번 기간 행만 교체. 연간 행은 month 빈 값
      → 미리 저장해 둔 다른 기간(연간·다른 달·다른 연도)은 그대로"""
    out = config.OUT_DIR / str(year)
    out.mkdir(parents=True, exist_ok=True)
    periods = [str(p) for p in periods]
    for name in list(tables):
        df = tables[name]
        if "period" in df.columns:
            df = df[df["period"].astype(str).isin(periods)]
        tables[name] = _year_month(df.reset_index(drop=True), year)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    if not to_db:
        return
    eng = connect()
    from sqlalchemy.types import Text
    for name, df in tables.items():
        key = ["year", "month"] if "month" in df.columns else ["year"]
        try:
            upsert(eng, name, df, key, {c: Text() for c in df.columns if c in LONG_TEXT})
        except Exception as e:  # noqa: BLE001
            raise JobError("E_DB_WRITE", f"{name} 저장 실패 ({db_errno(e)})") from e

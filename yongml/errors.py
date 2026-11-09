"""실행을 멈춰야 하는 오류. 로그·화면 출력은 없고, run_all.py 가 종료 코드 1로 끝남"""


class JobError(Exception):
    """code: E_DB_CONNECT 같은 짧은 오류 코드, detail: 짧은 사유"""

    def __init__(self, code, detail=""):
        super().__init__(f"{code} {detail}".strip())
        self.code, self.detail = code, detail


def db_errno(e):
    """DB 오류에서 번호만 (예: 1045 접속 거부, 2003 연결 안 됨, 1146 표 없음)"""
    o = getattr(e, "orig", None) or e
    a = getattr(o, "args", ())
    return a[0] if a and isinstance(a[0], int) else type(o).__name__

"""설정: DB 접속, LLM, 실무 기준, 추천·취약지역·지표 신뢰도 기준, 우선순위 기본 가중치"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- DB (yong 프로젝트와 같은 MariaDB)
DB = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT") or 3306),
    "name": os.getenv("DB_NAME", "yongin"),          # 데이터 DB (feat_ 읽기, ml_ 쓰기)
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
}
# 민원 테이블: 지금은 더미가 yongin.app_complaint 에 있음 → 나중에 'yongin_app.complaint' 로 바꾸면 됨
COMPLAINT_TABLE = os.getenv("COMPLAINT_TABLE", "app_complaint")

OUT_DIR = ROOT / "outputs"

# ---------------------------------------------------------------- LLM (설명 문장). 키가 없으면 템플릿 문장으로 대신함
_PROV = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
LLM = {
    "provider": _PROV,                                   # gemini | anthropic | openai
    "model": (os.getenv("LLM_MODEL") or "").strip(),     # 비우면 gemini 는 gemini-3.5-flash-lite (explain.DEFAULT_MODEL)
    "api_key": (os.getenv("LLM_API_KEY") or "").strip(),
    "rpm": float(os.getenv("LLM_RPM") or (10 if _PROV == "gemini" else 50)),  # 분당 최대 호출 (무료 등급은 천천히)
}
# LLM 서버 일시 오류(503 과부하 등 5xx, 연결 끊김) 때 기다렸다 다시 부르는 간격(초). 다 써도 안 되면 그 지역은 템플릿
LLM_RETRY_WAITS = [10, 30, 60]

# ---------------------------------------------------------------- 공통
MODES = {"BUS": "노선버스", "DRT": "똑버스", "HAPPY_TAXI": "행복택시"}
EUPMYEON = ["포곡읍", "모현읍", "이동읍", "남사읍", "원삼면", "백암면", "양지면"]
MVP_EMD = ["백암면", "원삼면"]          # 화면 MVP 대상 (학습은 읍면 리 전체)
# 기준연도: 여기 말고 run_all.py 맨 위 YEAR·MONTH 를 바꾸세요 (실행할 때 이 값을 덮어씀)
# 결과 표: year(2025) + month(1~12, 연간 행은 빈 값)
YEAR = "2025"
SEED = 42

# 민원을 적합성 점수에 섞는 비율 (0.15 = 모델·지표 85% + 민원 15%)
COMPLAINT_BLEND = 0.15
# 행복택시 점수에 '택시 호출 많고 잘 안 잡힘'(앱택시)을 섞는 비율. 모델 안에 넣지 않는 이유:
# 운영 마을 18개 중 16개가 백암·원삼이라 면 단위 값(앱택시)을 넣으면 모델이 '어느 면인지'만 외움
TAXI_BLEND = 0.15
# 'AI 추천 해결방안' 카드: 1위는 항상 추천 1. 2위는 이 점수 이상이고 1위와 차이가 이 안이면 추천 2, 아니면 제외
RECOMMEND_MIN = 50
RECOMMEND_GAP = 10
# 현행 유지: 지역 유형이 '대중교통 양호형'(뚜렷한 공백·혼잡 없음)이고 교통 취약지역(VULNERABLE_SCORE 이상)이 아니면
#   카드 맨 앞에 '현행 유지'가 추천 1, 세 수단은 모두 제외 → 카드 4개. 아니면 지금처럼 세 수단 카드 3개.
#   False 로 바꾸면 현행 유지 없이 모든 지역 3개
KEEP_CURRENT = True
# 근거로 쓰는 사실의 최소 차이: 이 지역 값이 평균과 이 비율 이상 달라야 '높다/낮다' 근거로 씀 (0.10 = 10%)
#   예) 택시 성사율 53% vs 평균 54% 는 근거로 안 씀. LLM 에도 '평균과 비슷'으로 넘김
FACT_MIN_GAP = 0.10
# 카드에 보이는 수단 설명 (이름, 부제, 한 줄 설명). KEEP = 현행 유지 카드
MODE_CARD = {
    "BUS": ("노선버스", "증차·노선 조정", "지금 다니는 버스를 더 자주, 더 가까이 다니게 해요"),
    "DRT": ("똑버스", "수요응답형 버스", "정해진 노선 없이 호출에 따라 운행해요"),
    "HAPPY_TAXI": ("행복택시", "교통 소외지역 택시", "정해진 요금으로 마을과 거점을 이어요"),
    "KEEP": ("현행 유지", "지금 수단 유지", "새 수단 없이 지금 운영을 그대로 이어가요"),
}
# 점수가 이 이상이면 'AI 추천'으로 표시
AI_RECOMMEND_SCORE = 70

# ---------------------------------------------------------------- 실무 기준 (비교용, 정답 아님)
RULES = {
    "happy_runs": 5,          # 하루 버스 5회 이하 → 행복택시 (실무자 통화)
    "happy_dong_share": 0.2,  # 행정동 단위: 5회 이하 정류장이 20% 이상
    "bus_density_pct": 0.7,   # 인구밀도 상위 30% → 노선버스 증차·조정
    "drt_density_pct": 0.5,   # 인구밀도 하위 50% 이면서 흩어짐 → 똑버스
}

# ---------------------------------------------------------------- 대시보드 '교통 취약지역': 종합 취약도(기본 가중치)가 이 점수 이상
VULNERABLE_SCORE = 50
# ---------------------------------------------------------------- 지표 신뢰도 낮음 (화면 경고 문구). 기준값은 협의 후 바꾸기
RELIABILITY_MIN_CALLS = 100   # 월평균 앱택시 호출이 이보다 적으면 (리는 그 면 값)
RELIABILITY_MIN_POP = 300     # 리 인구가 이보다 적으면 (비율 지표가 크게 흔들림)

# ---------------------------------------------------------------- 우선순위 평가 기본 가중치 (Spring 이 화면에서 바꿀 수 있음)
# 처음 25/20/20/15/10/5/5 → 2025 데이터로 점검 후 30/20/20/10/10/5/5 로 조정
#  - 부적합은 '점수 차이'라 대부분 0 근처(표준편차 15, 다른 항목 23~29) → 25 를 줘도 실제 영향은 15~18% 뿐이었음
#  - 취약인구는 서비스 부족·민원과 겹쳐서(상관 0.4~0.5) 15 를 줘도 리에서 실제 영향 28%
#  → 조정 후 실제 영향: 부적합 22~23%, 접근성 23~29%, 서비스 23~29%, 취약인구 10~18%, 민원 15~17%
PRIORITY_WEIGHTS = {
    "mismatch": 30,     # 교통수단 부적합
    "access": 20,       # 접근성 취약
    "service": 20,      # 서비스 수준 부족
    "vulnerable": 10,   # 취약인구
    "complaint": 10,    # 민원
    "demand": 5,        # 잠재 수요
    "feasibility": 5,   # 실행 가능성
}

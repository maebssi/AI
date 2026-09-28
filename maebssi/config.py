"""프로젝트 공통 경로·상수 설정.

원천 데이터 위치는 환경변수 MAEBSSI_RAW_DIR 로 바꿀 수 있다.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = Path(os.environ.get(
    "MAEBSSI_RAW_DIR",
    r"D:\67.제조현장 이송장치의 열화 예지보전 멀티모달 데이터\3.개방데이터\1.데이터",
))

# 추가 공개 데이터(AI4I, KOSHA Guide, AssetOpsBench) 저장 위치 — scripts/download_data.py
EXTERNAL_DATA_DIR = Path(os.environ.get("MAEBSSI_EXTERNAL_DIR", r"D:\maebssi_data"))
AI4I_CSV = EXTERNAL_DATA_DIR / "ai4i" / "ai4i2020.csv"
KOSHA_DIR = EXTERNAL_DATA_DIR / "kosha"
ASSETOPS_HF_DIR = EXTERNAL_DATA_DIR / "AssetOpsBench_hf"
MANUAL_DIR = EXTERNAL_DATA_DIR / "manuals"
# 자동 다운로드 대상 KOSHA Guide (2021년 이후 개정본만 공단 엔드포인트에서 제공)
KOSHA_GUIDES = {
    "Z-30-2022": "제조업 등의 정비보수 절차에 관한 지침",
    "Z-6-2022": "작업장 안전확인 및 점검에 관한 지침",
    "Z-28-2022": "안전보건표지 설치 및 유지관리에 관한 지침",
    "G-10-2023": "작업장 내 운반차량의 운행에 관한 안전가이드",
    "M-137-2023": "기계의 제작·구매·사용 시 안전기준에 관한 기술지침",
}

ARTIFACT_DIR = Path(os.environ.get("MAEBSSI_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
PROCESSED_DIR = ARTIFACT_DIR / "processed"
MODEL_DIR = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"
DB_PATH = ARTIFACT_DIR / "plant.db"
LOG_DIR = ARTIFACT_DIR / "logs"

KNOWLEDGE_DOC_DIR = PROJECT_ROOT / "maebssi" / "knowledge" / "docs"
EMBEDDING_MODEL = os.environ.get("MAEBSSI_EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")

SENSORS = ["NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4"]
EXTERNAL = ["ex_temperature", "ex_humidity", "ex_illuminance"]

STATE_NAMES = {0: "정상", 1: "관심", 2: "경고", 3: "위험"}
STATE_CODES = {0: "NORMAL", 1: "ATTENTION", 2: "WARNING", 3: "DANGER"}

THERMAL_SHAPE = (120, 160)
THUMB_SHAPE = (30, 40)  # 열화상 4x4 평균 풀링 썸네일 (CNN 입력)

for d in (PROCESSED_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

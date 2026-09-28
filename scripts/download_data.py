"""과제 수행계획서 3.3 의 공개 데이터 다운로드.

로그인 없이 받을 수 있는 데이터만 자동으로 받는다. 이미 있는 파일은 건너뛴다.
  python -m scripts.download_data            # 전체
  python -m scripts.download_data ai4i kosha # 일부

회원가입·이용 승인이 필요한 데이터(KAMP, AI허브)는 --status 로 받는 방법과 둘 위치를 안내한다.
"""
import io
import subprocess
import sys
import time
import urllib.request
import zipfile

from maebssi.config import AI4I_CSV, ASSETOPS_HF_DIR, EXTERNAL_DATA_DIR, KOSHA_DIR, KOSHA_GUIDES, MANUAL_DIR, RAW_DIR

UA = {"User-Agent": "maebssi-research/0.1 (SW project; data download)"}

KOSHA_URL = "https://oshri.kosha.or.kr/extappKosha/kosha/guidance/fileDownload.do?sfhlhTchnlgyManualNo={}&fileOrdrNo={}"

ASSETOPS_HF_FILES = [
    "README.md",
    "data/scenarios/all_utterance.jsonl",
    "data/task/failure_mapping_senarios.jsonl",
    "data/task/rule_monitoring_scenarios.jsonl",
    "data/failuresensoriq_standard/all.jsonl",
    "data/failuresensoriq_standard/sample_50_questions.jsonl",
]
HF_URL = "https://huggingface.co/datasets/ibm-research/AssetOpsBench/resolve/main/{}"

MANUAL = [
    ("KAMP 제조AI 데이터셋", "https://www.kamp-ai.kr → 회원가입 후 'AI 데이터셋'에서 필요한 업종 데이터셋(예: 사출성형) 다운로드",
     EXTERNAL_DATA_DIR / "kamp", "CSV 를 넣은 뒤 python -m maebssi.analytics.db --add <csv경로> <테이블명> 으로 분석 DB에 등록"),
    ("AI허브 기계시설물 고장 예지 센서 (dataSetSn=238)", "https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=238 → 로그인·이용 신청·승인 후 다운로드",
     EXTERNAL_DATA_DIR / "aihub_motor", "전동기 진동·전류 데이터. 받은 뒤 스키마 확인 후 로더 추가 예정"),
    ("설비 제조사 공개 매뉴얼", "제조사 홈페이지(제품별 상이)",
     MANUAL_DIR, "PDF 를 넣으면 지식 에이전트가 자동 색인"),
    ("KOSHA E-91 등 2021년 이전 지침", "https://www.kosha.or.kr → 자료마당 → KOSHA Guide 에서 직접 다운로드",
     KOSHA_DIR, "KOSHA_<번호>.pdf 형식으로 저장하면 자동 색인"),
]


def fetch(url: str, timeout=120) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def ai4i():
    if AI4I_CSV.exists():
        return print("[ai4i] 이미 있음", AI4I_CSV)
    AI4I_CSV.parent.mkdir(parents=True, exist_ok=True)
    data = fetch("https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip")
    zipfile.ZipFile(io.BytesIO(data)).extractall(AI4I_CSV.parent)
    print("[ai4i] 저장", AI4I_CSV)


def kosha():
    KOSHA_DIR.mkdir(parents=True, exist_ok=True)
    for code, title in KOSHA_GUIDES.items():
        out = KOSHA_DIR / f"KOSHA_{code}.pdf"
        if out.exists():
            print("[kosha] 이미 있음", out.name)
            continue
        for n in (2, 1, 3):  # 첨부 순번: 보통 2번이 본문 PDF
            try:
                b = fetch(KOSHA_URL.format(code, n), 60)
            except Exception:
                b = b""
            time.sleep(0.3)
            if b[:4] == b"%PDF":
                out.write_bytes(b)
                print("[kosha] 저장", out.name, title)
                break
        else:
            print("[kosha] 실패", code, title)


def assetops():
    for f in ASSETOPS_HF_FILES:
        out = ASSETOPS_HF_DIR / f
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(fetch(HF_URL.format(f)))
        print("[assetops] 저장", f)
    repo = EXTERNAL_DATA_DIR / "AssetOpsBench"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/IBM/AssetOpsBench", str(repo)], check=True)
        print("[assetops] 저장소 클론", repo)


def status():
    print("원천(AI허브 이송장치):", "있음" if RAW_DIR.exists() else "없음", RAW_DIR)
    print("AI4I:", "있음" if AI4I_CSV.exists() else "없음")
    print("KOSHA PDF:", len(list(KOSHA_DIR.glob("KOSHA_*.pdf"))) if KOSHA_DIR.exists() else 0, "개")
    print("AssetOpsBench(HF):", "있음" if (ASSETOPS_HF_DIR / "data").exists() else "없음")
    print("FAB-Bench: 논문에 적힌 github.com/FuturefabAI/FAB-Bench 가 현재 404 — 공개 후 추가")
    print("\n직접 받아야 하는 데이터 (회원가입·승인 필요):")
    for name, how, where, note in MANUAL:
        print(f"- {name}\n    받는 곳: {how}\n    둘 위치: {where}\n    비고: {note}")


TASKS = {"ai4i": ai4i, "kosha": kosha, "assetops": assetops}

if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
        sys.exit()
    for name in [a for a in sys.argv[1:] if not a.startswith("-")] or list(TASKS):
        TASKS[name]()
    status()

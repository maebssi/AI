# 맵씨 AI — 제조 Agentic AI 기반 스마트공장 운영 지원

2026 SW 프로젝트 과제 수행계획서의 **설비 상태 → 설비 지식(RAG) → 작업지시 초안 → 관리자 승인** 흐름을
AI허브 「제조현장 이송장치의 열화 예지보전 멀티모달 데이터」(OHT·AGV 센서 8종 + 열화상, 4단계 상태)로 구현한 프로토타입.
계획서 3.3의 공개 데이터 중 로그인 없이 받을 수 있는 AI4I 2020·KOSHA Guide·AssetOpsBench로 생산·품질 분석 에이전트, 공공 지침 RAG, 고장모드 지식그래프를 추가했다.

```
             ┌──────────── LangGraph 오케스트레이터 (maebssi/orchestrator/graph.py) ────────────┐
 센서+열화상 ─▶ monitor ─┬─(정상)──▶ 종료                                                         │
 (1초 주기)   설비 상태   ├─(관심)──▶ watch: 모니터링 강화                                          │
             에이전트    └─(경고·위험)▶ diagnose ─▶ plan ─────────────▶ draft ─▶ approval ⏸ ─▶ 발행 │
                                     설비 지식    정비이력·부품재고·     작업지시   관리자 승인       │
                                     에이전트(RAG) 기술자·생산영향        초안       (P1은 2인)       │
             └──────────────────────── 모든 단계 audit_log 기록 ─────────────────────────────────┘
 관리자 질의 ─▶ qa.route ─▶ 설비 상태 / 작업지시 / 부품 재고 / 생산·품질 데이터(NL→SQL) / 지식 RAG(근거 없으면 거절)
 diagnose 는 RAG(팀 샘플 문서 + KOSHA Guide) 와 고장모드 지식그래프(AssetOpsBench)를 함께 조회
```

## 계획서 항목 대응

| 계획서 (2.1 / 3.1) | 구현 | 위치 |
|---|---|---|
| ③ 설비 이상 탐지 (시계열+이미지 멀티모달), 위험도 4단계 | 센서·열화상 통계·인과적 롤링 특징 GBM + 열화상 CNN·특징 MLP 융합 신경망 앙상블, 정상 구간 IsolationForest 이상점수 | `maebssi/equipment/` |
| ① 제조 RAG (문서 파싱·청킹·한국어 임베딩·하이브리드 검색·인용·거절) | 마크다운·KOSHA PDF 절 단위 청킹, BM25 + ko-sroberta 임베딩 RRF 결합, 쪽 번호까지 인용, 근거 부족 시 거절 | `maebssi/knowledge/rag.py`, `pdf_loader.py` |
| ① 확장: 설비–고장–센서 지식그래프 (GraphRAG) | AssetOpsBench FailureSensorIQ 2,667문항에서 관계 1,306개 추출, 증상 → 전동기 고장모드 후보 | `maebssi/knowledge/failure_graph.py` |
| ④ 생산·품질 분석 (자연어→SQL, 불량 예측, 원인 변수) | AI4I 2020: 규칙 기반 의미 파서(LLM 있으면 LLM 우선) + 읽기 전용 SQL, GBM 고장 예측, 가림 기법 원인 변수와 권장 조치 | `maebssi/analytics/` |
| ② Agentic 오케스트레이션, 승인 게이트, 행동 로그 | LangGraph StateGraph + `interrupt` 승인 게이트, 감사 로그 | `maebssi/orchestrator/` |
| 작업지시·부품 재고 모의 DB | SQLite: 설비 36대, 부품, 기술자·일정, 정비 이력, 시간대별 반송 계획, 작업지시, 감사 로그 | `maebssi/workorder/` |
| FastAPI 추론 서버, MCP 도구 연동 | REST API + 시연 대시보드, MCP 도구 서버 | `server/` |
| 에이전트별 정량 평가, 시나리오 성공률 | 분류 F1·재현율, Recall@5·인용 정확도·거절 정확도, 시나리오 성공률 | `scripts/evaluate.py`, `eval/` |

LLM은 선택 사항이다. `anthropic` 패키지와 `ANTHROPIC_API_KEY`가 있으면 Claude(`claude-opus-5`)가 RAG 답변과 작업지시 문장을 다듬고,
없으면 추출식 답변과 템플릿으로 동일하게 동작한다(평가는 재현성을 위해 LLM 끔).

## 실행

```bash
pip install -r requirements.txt
# 0) 공개 데이터 다운로드 (AI4I, KOSHA Guide 5종, AssetOpsBench → D:\maebssi_data) / 로그인 필요 데이터 안내
python -m scripts.download_data
# 1) 전처리 (원천 zip 5.2GB → artifacts/processed, 약 1~2분 @12 workers)
python -m maebssi.data.build_dataset --workers 12
# 2) 설비 상태 모델 학습·평가 (GPU 약 1분)
python -m maebssi.equipment.train
# 3) 모의 DB·분석 DB 생성, 불량 예측 모델 학습
python -m maebssi.workorder.db --reset
python -m maebssi.analytics.db
python -m maebssi.analytics.quality
# 4) 정량 평가 → artifacts/reports/evaluation.json
python -m scripts.evaluate
# 5) 서버 + 대시보드 → http://localhost:8000  (API 문서 /docs)
uvicorn server.app:app --port 8000
```

추가 데이터 위치는 `MAEBSSI_EXTERNAL_DIR`(기본 `D:\maebssi_data`)로 바꿀 수 있다. 원천 데이터 경로 기본값은 `D:\67.제조현장 이송장치의 열화 예지보전 멀티모달 데이터\3.개방데이터\1.데이터`이며 `MAEBSSI_RAW_DIR`로 바꿀 수 있다.
산출물(`artifacts/`)은 git에 올리지 않는다.

## 데이터

계획서 3.3 데이터별 현황 (`python -m scripts.download_data --status`)

| 데이터 | 상태 | 쓰는 곳 |
|---|---|---|
| AI허브 이송장치 열화 예지보전 | D드라이브 보유 | 설비 상태 에이전트 |
| UCI AI4I 2020 (CC BY 4.0) | **자동 다운로드** | 생산·품질 분석 에이전트 |
| KOSHA Guide | **자동 다운로드 5종** (Z-30 정비보수 절차, Z-6 안전점검, Z-28 안전표지, G-10 운반차량, M-137 기계 안전기준) | RAG 코퍼스 (134 청크) |
| IBM AssetOpsBench (Apache-2.0) | **자동 다운로드** (GitHub 코드·샘플 + HuggingFace 시나리오 152개·FailureSensorIQ 2,667문항) | 고장모드 지식그래프, 평가 설계 참고 |
| FAB-Bench | 받을 수 없음 — 논문의 `github.com/FuturefabAI/FAB-Bench`가 현재 404 | 공개되면 RAG 평가에 추가 |
| KAMP 제조AI 데이터셋 | 직접 받아야 함 (회원가입) | `python -m maebssi.analytics.db --add <csv> <테이블>` 로 분석 DB에 등록 |
| AI허브 기계시설물 고장 예지 센서 | 직접 받아야 함 (이용 신청·승인) | 받은 뒤 로더 추가 |
| KOSHA E-91(LOTO) 등 2021년 이전 지침, 제조사 매뉴얼 | 직접 받아야 함 (공단 엔드포인트는 2021년 이후 개정본만 제공) | `D:\maebssi_data\kosha`, `manuals`에 PDF를 두면 자동 색인 |

**이송장치 데이터 분할**

| split | 세션 | 프레임 | 설비 | 정상 / 관심 / 경고 / 위험 |
|---|---|---|---|---|
| Training | 303 | 99,476 | agv·oht 01~16 | 49,285 / 21,189 / 21,322 / 7,680 |
| Validation | 38 | 12,394 | agv·oht **17~18 (학습에 없는 설비)** | 5,643 / 2,892 / 2,869 / 990 |

한 세션(약 5~10분, 1초 간격)은 정상에서 위험으로 열화가 진행되는 구조이다. 따라서 세션 경과 시간처럼 라벨을 누설하는 특징은 쓰지 않았고,
과거 프레임만 사용하는 롤링 평균·표준편차·기울기 특징만 썼다(`tests/test_units.py::test_features_are_causal`에서 검증).

## 결과 (Validation, 학습에 없는 설비)

**설비 상태 에이전트** — 프레임 단위 4단계 분류

| 모델 | macro F1 | 정확도 | 재현율 정상/관심/경고/위험 | 경고 이상 재현율 | 정상 오경보율 |
|---|---|---|---|---|---|
| GBM (특징 115개) | 0.961 | 0.963 | 0.960 / 0.951 / 0.978 / 0.977 | 0.987 | 2.5% |
| 열화상 CNN + MLP 융합 | 0.928 | 0.925 | 0.916 / 0.893 / 0.952 / 0.989 | 0.974 | 2.6% |
| **앙상블 (서비스 기본)** | **0.962** | **0.963** | 0.957 / 0.956 / 0.975 / 0.982 | **0.985** | **2.4%** |

- 설비 종류별 앙상블 macro F1: OHT 0.975, AGV 0.940
- 5초 간격 스트리밍 감지: 경고가 있는 33개 세션 모두 감지, 감지 지연 중앙값 4초(7개 세션은 라벨보다 먼저 감지), 경고가 없는 5개 세션 중 1개에서 오경보(agv18_1026_2356, 최대 라벨 '관심')
- 이상점수 평균: 정상 0.41 → 관심 0.60 → 경고 0.70 → 위험 0.73

**설비 지식 에이전트 (RAG)** — 코퍼스: 팀 샘플 문서 34 청크 + KOSHA Guide 134 청크

| 평가셋 | Recall@5 | MRR@5 | 범위 밖 질문 거절 | 잘못된 거절 | 인용 정밀도(문서 단위) |
|---|---|---|---|---|---|
| `rag_eval` (답변 가능 42 + 범위 밖 8, 임계값 보정에 사용) | 1.00 | 0.952 | 7/8 | 0% | 0.85 |
| `rag_holdout` (답변 가능 11 + 범위 밖 8, 보정에 미사용) | 1.00 | 1.00 | **6/8** | 0% | 0.95 |

KOSHA 문서를 넣자 코퍼스 어휘가 늘어 기존 거절 기준(코퍼스 전체 어휘 대비 질문 적합도)으로는 범위 밖 거절이 4/8로 떨어졌다. 그래서 적합도를 **검색된 근거 청크 기준**으로 바꿨다. 보정에 쓰지 않은 검증셋에서도 여전히 2건(엘리베이터 와이어로프, 연차 규정)을 거절하지 못한다.

**생산·품질 분석 에이전트 (AI4I 2020, 1만 건, 고장 339건)**

| 항목 | 결과 |
|---|---|
| 고장 예측 AUC (층화 5-fold) | **0.975** ± 0.004 (PR-AUC 0.88) — 로지스틱 기준선 0.928, 파생변수 없는 GBM 0.972 (PR-AUC 0.80) |
| 홀드아웃 20% | AUC 0.982, 재현율 0.82 / 정밀도 0.92 (임계값 0.5) |
| 원인 변수 1순위 적중률 | **91%** (60/66) — 실제 고장 유형을 규정하는 변수군에 1순위 원인이 속하는 비율. 놓친 건은 모델이 확률을 거의 주지 못한 공구마모 고장(TWF) |
| 자연어→SQL 실행 정확도 (규칙 파서) | 개발용 셋 25/25, **별도 검증셋 14/15 (93%)** — 실패: "최소값과 최대값"처럼 한 질문에 집계가 둘 |

**고장모드 지식그래프 (AssetOpsBench FailureSensorIQ)**

- 문항 무작위 분할 정확도 1.00: 같은 사실이 문장만 바꿔 반복 출제되어 조회로 맞힘 → 그래프가 사실을 빠짐없이 담았다는 뜻일 뿐 추론 능력 지표는 아님
- **처음 보는 고장모드** (그 고장모드가 나오는 모든 문항을 학습에서 제외): 262문항 정확도 **0.59** (무작위 선택 0.20)

**통합 시나리오** — 위험까지 진행되는 Validation 33개 세션 × 각 단계 진입 20초 후 = 132건
성공 기준: 위험도 분기 정확 + (경고·위험이면) 올바른 우선순위(P2/P1)의 초안, 문서 인용 포함, 승인 전 미발행, 승인 후 발행. **성공률 100% (132/132)**.
관리자 질의 라우팅 정확도 100% (19/19).

## 한계와 다음 단계

- **OHT·AGV 정비 문서는 여전히 팀 작성 샘플이다** (`maebssi/knowledge/docs`, 부품번호·기준값 가상). KOSHA 지침은 실제 공공 문서지만 일반 안전·정비 절차라, 설비별 매뉴얼은 제조사 문서로 채워야 한다. LOTO 전용 지침(E-91-2016)은 자동으로 받을 수 없어 샘플 문서가 대신한다.
- **RAG 평가셋이 작다** (검증셋 19문항). 범위 밖 질문 거절이 6/8에 그치며, 리랭커(cross-encoder)는 아직 넣지 않았다.
- **AI4I는 합성 데이터이며 OHT·AGV와 다른 가공 설비다.** 생산·품질 분석 에이전트의 구조(자연어→SQL, 예측, 원인 변수)를 검증하는 용도이고, KAMP 실데이터를 받으면 `--add`로 테이블을 등록해 같은 도구로 조회할 수 있다. 다만 규칙 파서는 AI4I 열 동의어에 맞춰져 있어, 다른 테이블 조회는 LLM 경로가 필요하다.
- **지식그래프는 영어 어휘(electric motor 등)이며**, 이송장치 신호를 전동기 센서(current·power·temperature 등)에 수동으로 매핑했다.
- **시나리오 성공률 100%는 라벨 단계가 뚜렷한 시점(진입 20초 후)에서 측정한 값이다.** 단계 경계 직후에는 판정이 흔들릴 수 있으며, 최근 5프레임 확률 평균으로 완화하고 있다.
- 오케스트레이터 체크포인트가 메모리(`InMemorySaver`)라 서버를 재시작하면 진행 중인 승인 대기 스레드가 사라진다(작업지시 초안은 DB에 남음).
- MCP 서버(`server/mcp_server.py`)는 `pip install mcp` 후 사용 가능하며, 이 환경에서는 실행 검증을 하지 않았다.

## 구조

```
maebssi/
  config.py                 경로·상수
  llm.py                    선택적 Claude 호출 (없으면 None → 대체 로직)
  data/build_dataset.py     원천 zip → frames.parquet + 열화상 썸네일
  equipment/                features.py · nets.py · train.py · agent.py(추론·스트림 재생)
  knowledge/                rag.py · pdf_loader.py(KOSHA·매뉴얼 PDF) · failure_graph.py(지식그래프) · docs/*.md
  analytics/                db.py(분석 DB, CSV 등록) · nl2sql.py(자연어→SQL) · quality.py(불량 예측·원인 변수)
  workorder/                db.py(모의 DB) · tools.py(이력·재고·기술자·생산영향·초안·승인)
  orchestrator/             graph.py(LangGraph 이상 대응) · qa.py(관리자 질의 라우팅)
server/                     app.py(FastAPI) · static/index.html(대시보드) · mcp_server.py
scripts/download_data.py    공개 데이터 다운로드·현황
scripts/evaluate.py         정량 평가
eval/                       RAG·NL2SQL(개발/검증)·라우팅 평가셋
tests/                      단위 테스트 (python -m pytest tests)
```

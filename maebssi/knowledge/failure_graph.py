"""설비–고장모드–센서 지식그래프 (계획서 3.1 ① 확장: GraphRAG).

출처: IBM AssetOpsBench / FailureSensorIQ (Apache-2.0, 2,667 객관식 문항, 설비 10종).
각 문항의 정답에서 관계를 뽑는다.
  - relevant_*   문항: 정답 (고장모드, 센서) → +1
  - irrelevant_* 문항: 정답 → −1, 나머지 보기 → +0.3 (정답이 아닌 보기는 관련 있을 가능성이 큼)
문항 문장은 템플릿이 다양하므로, 보기에서 모은 설비별 고장모드·센서 어휘가 질문에 등장하는지로 대상을 찾는다.

평가 (python -m maebssi.knowledge.failure_graph):
  - random  : 문항 80/20 무작위 분할 — 그래프의 사실 재현 능력
  - grouped : (설비, 고장모드|센서) 묶음 단위 분할 — 처음 보는 조합에 대한 일반화(설비 내 사전빈도로만 추론)
"""
import json
import random
from collections import defaultdict
from functools import lru_cache

from maebssi.config import ASSETOPS_HF_DIR, REPORT_DIR

FSIQ = ASSETOPS_HF_DIR / "data" / "failuresensoriq_standard" / "all.jsonl"
SENSOR_Q = {"relevant_sensors_for_failure_mode", "irrelevant_sensors_for_failure_mode"}
POSITIVE = {"relevant_sensors_for_failure_mode", "relevant_failure_modes_for_sensor"}

# 이송장치(OHT·AGV) 신호 → FailureSensorIQ 의 전동기(electric motor) 센서 어휘
SIGNAL_TO_SENSOR = {"과전류": ["current", "power"], "과열": ["temperature", "winding temperature"],
                    "분진": ["vibration"]}


def load_questions():
    return [json.loads(l) for l in FSIQ.read_text("utf8").splitlines() if l.strip()]


def vocab(questions):
    fm, sensors = defaultdict(set), defaultdict(set)
    for q in questions:
        target = sensors if q["relevancy"] in SENSOR_Q else fm
        target[q["asset_name"]].update(o.strip().lower() for o in q["options"])
    return fm, sensors


def subject_of(q, fm_vocab, sensor_vocab):
    """문항이 묻는 대상(센서 문항이면 고장모드, 고장모드 문항이면 센서)을 질문 문장에서 찾음 — 가장 긴 일치."""
    text = q["question"].lower()
    pool = fm_vocab[q["asset_name"]] if q["relevancy"] in SENSOR_Q else sensor_vocab[q["asset_name"]]
    hits = [v for v in pool if v and v in text]
    return max(hits, key=len) if hits else None


class FailureGraph:
    def __init__(self, questions):
        self.fm_vocab, self.sensor_vocab = vocab(questions)
        self.edge = defaultdict(float)  # (asset, failure_mode, sensor) → 점수
        self.prior = defaultdict(float)  # (asset, 보기) → 설비 내 관련 빈도 (처음 보는 조합용)
        self.unparsed = 0
        for q in questions:
            self._add(q)

    def _key(self, q, subj, opt):
        a = q["asset_name"]
        return (a, subj, opt) if q["relevancy"] in SENSOR_Q else (a, opt, subj)

    def _add(self, q):
        subj = subject_of(q, self.fm_vocab, self.sensor_vocab)
        if subj is None:
            self.unparsed += 1
            return
        pos = q["relevancy"] in POSITIVE
        for opt, ok in zip(q["options"], q["correct"]):
            opt = opt.strip().lower()
            if pos and ok:
                w = 1.0
            elif not pos:
                w = -1.0 if ok else 0.3
            else:
                continue
            self.edge[self._key(q, subj, opt)] += w
            self.prior[(q["asset_name"], opt)] += w

    def answer(self, q) -> int:
        """객관식 문항에 대한 보기 번호 예측."""
        subj = subject_of(q, self.fm_vocab, self.sensor_vocab)
        scores = []
        for opt in q["options"]:
            o = opt.strip().lower()
            s = self.edge.get(self._key(q, subj, o), 0.0) if subj else 0.0
            scores.append(s + 0.01 * self.prior.get((q["asset_name"], o), 0.0))
        pick = max if q["relevancy"] in POSITIVE else min
        return scores.index(pick(scores))

    # ── 서비스용 조회 ──
    def sensors_for_failure(self, asset, failure_mode, k=5):
        rows = [(s, w) for (a, f, s), w in self.edge.items() if a == asset and f == failure_mode and w > 0]
        return sorted(rows, key=lambda x: -x[1])[:k]

    def failures_for_sensor(self, asset, sensor, k=5):
        rows = [(f, w) for (a, f, s), w in self.edge.items() if a == asset and s == sensor and w > 0]
        return sorted(rows, key=lambda x: -x[1])[:k]


@lru_cache(maxsize=1)
def get_graph() -> FailureGraph | None:
    if not FSIQ.exists():
        return None
    return FailureGraph(load_questions())


def failure_modes_for_symptoms(tags: list[str], asset: str = "electric motor", k: int = 4) -> list[dict]:
    """이송장치 증상 태그 → 전동기 고장모드 후보 (지식그래프 근거)."""
    g = get_graph()
    if g is None:
        return []
    agg, via = defaultdict(float), defaultdict(list)
    for t in tags:
        for sensor in SIGNAL_TO_SENSOR.get(t, []):
            for fm, w in g.failures_for_sensor(asset, sensor, 10):
                agg[fm] += w
                if sensor not in via[fm]:
                    via[fm].append(sensor)
    rows = sorted(agg.items(), key=lambda x: -x[1])[:k]
    return [{"failure_mode": fm, "sensor": "/".join(via[fm]), "weight": round(w, 2),
             "source": "[AssetOpsBench FailureSensorIQ]"} for fm, w in rows]


def evaluate(seed=42) -> dict:
    qs = load_questions()
    rnd = random.Random(seed)
    idx = list(range(len(qs)))
    rnd.shuffle(idx)
    cut = int(len(idx) * 0.8)
    out = {"n_questions": len(qs), "chance": 0.2}

    def acc(train, test):
        g = FailureGraph(train)
        return sum(g.answer(q) == q["correct"].index(True) for q in test) / len(test), g

    a, g = acc([qs[i] for i in idx[:cut]], [qs[i] for i in idx[cut:]])
    out["random_split_acc"] = a
    out["edges"] = len(g.edge)
    out["unparsed_rate"] = g.unparsed / cut

    # 고장모드 보류 분할: 보류 고장모드가 질문 대상이거나 보기에 등장하는 문항은 학습에서 모두 제외
    # (역방향 문항으로 같은 사실이 새지 않게) → 보류 고장모드의 센서 문항으로 평가
    fm_v, s_v = vocab(qs)
    pairs = sorted({(a, f) for a, fs in fm_v.items() for f in fs})
    rnd.shuffle(pairs)
    held = set(pairs[: len(pairs) // 5])

    def touches(q):
        a = q["asset_name"]
        if q["relevancy"] in SENSOR_Q:
            return (a, subject_of(q, fm_v, s_v)) in held
        return any((a, o.strip().lower()) in held for o in q["options"])

    train = [q for q in qs if not touches(q)]
    test = [q for q in qs if q["relevancy"] in SENSOR_Q and touches(q)]
    out["heldout_failure_mode_acc"], _ = acc(train, test)
    out["heldout_n_test"] = len(test)
    for split in ("positive", "negative"):
        sub = [qs[i] for i in idx[cut:] if (qs[i]["relevancy"] in POSITIVE) == (split == "positive")]
        out[f"random_split_acc_{split}"] = sum(g.answer(q) == q["correct"].index(True) for q in sub) / len(sub)
    return out


if __name__ == "__main__":
    r = evaluate()
    print(json.dumps(r, indent=1))
    (REPORT_DIR / "failure_graph_eval.json").write_text(json.dumps(r, indent=1), "utf8")
    g = get_graph()
    print(sorted(g.sensor_vocab["electric motor"]))
    for t in (["과전류"], ["과열"], ["과열", "과전류"]):
        print(t, failure_modes_for_symptoms(t))

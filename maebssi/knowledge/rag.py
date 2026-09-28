"""설비 지식 에이전트 — 하이브리드 검색(BM25 + 한국어 임베딩) RAG.

1) 문서 파싱·청킹: 마크다운은 '## 절' 단위, KOSHA Guide·매뉴얼 PDF 는 번호 절 단위(pdf_loader)로 나누고
   문서 제목을 앞에 붙인다.
2) 색인: BM25(형태 무관 토큰 + 한글 문자 bigram) / ko-sroberta 임베딩(코사인)
3) 검색: 두 순위를 RRF(Reciprocal Rank Fusion)로 결합
4) 답변: 근거 청크 인용([문서번호 §절]) 형태로 생성, 근거가 약하면 답변 거절
   - LLM 가능 시 Claude 로 생성, 아니면 질의와 겹치는 문장을 추출해 요약
"""
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from maebssi import llm
from maebssi.config import EMBEDDING_MODEL, KNOWLEDGE_DOC_DIR, KOSHA_DIR, MANUAL_DIR, MODEL_DIR

REFUSAL = "제공된 설비 지식 문서에서 근거를 찾을 수 없어 답변할 수 없습니다. 담당 엔지니어에게 문의하세요."
# 거절 임계값 — eval/rag_eval.jsonl 로 보정 (eval/rag_holdout.jsonl 로 별도 검증)
DENSE_MIN = 0.42
COVERAGE_MIN = 0.38


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    doc_no: str
    title: str
    section: str
    text: str
    page: int | None = None
    source: str = "sample"  # sample(팀 작성 샘플) / kosha / manual

    @property
    def cite(self) -> str:
        return f"[{self.doc_no} §{self.section}" + (f" p.{self.page}]" if self.page else "]")


def load_chunks() -> list[Chunk]:
    chunks = []
    for path in sorted(KNOWLEDGE_DOC_DIR.glob("*.md")):
        if path.name.startswith("00_"):
            continue
        raw = path.read_text("utf8")
        title = re.search(r"^# (.+)$", raw, re.M).group(1).strip()
        m = re.search(r"문서번호:\s*([\w-]+)", raw)
        doc_no = m.group(1) if m else path.stem
        for sec in re.split(r"^## ", raw, flags=re.M)[1:]:
            head, _, body = sec.partition("\n")
            body = body.strip()
            if body:
                chunks.append(Chunk(f"{path.stem}#{head.strip()}", path.stem, doc_no, title,
                                    head.strip(), f"{title} - {head.strip()}\n{body}"))
    chunks += load_pdf_chunks()
    return chunks


def load_pdf_chunks() -> list[Chunk]:
    """외부 데이터 폴더의 KOSHA Guide·제조사 매뉴얼 PDF (폴더가 없으면 건너뜀)."""
    from maebssi.knowledge.pdf_loader import load_pdf
    chunks = []
    for folder, source in ((KOSHA_DIR, "kosha"), (MANUAL_DIR, "manual")):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.pdf")):
            if source == "kosha" and not path.stem.startswith("KOSHA_"):
                continue  # 지침 목록 PDF 등 제외
            try:
                doc_no, title, parts = load_pdf(path)
            except Exception as e:
                print(f"[knowledge] PDF 건너뜀 {path.name}: {e}")
                continue
            for i, c in enumerate(parts):
                chunks.append(Chunk(f"{path.stem}#{c['section']}#{i}", path.stem, doc_no, title, c["section"],
                                    f"{title} - {c['section']}\n{c['text']}", c["page"], source))
    return chunks


_TOKEN = re.compile(r"[A-Za-z]+[\w.\-]*|\d+(?:\.\d+)?|[가-힣]+")
# 질문 형식어 — 질의 적합도(coverage) 계산에서 제외
_QUESTION_WORDS = {"알려줘", "알려", "주세요", "뭐야", "뭐가", "무엇", "무엇을", "무엇인가", "무엇인가요", "어떻게",
                   "하나", "하나요", "해야", "해야해", "인가", "인가요", "있나", "있나요", "얼마", "얼마나", "몇", "은", "는",
                   "나요", "까요", "좀", "순서대로", "정도", "해", "어떤", "언제", "왜"}


def query_coverage(query: str, vocab: dict) -> float:
    """질의의 내용어 중 코퍼스에 등장하는 비율 (한글 단어는 bigram 절반 이상이 있으면 등장으로 간주)."""
    words = [w for w in _TOKEN.findall(query.lower()) if w not in _QUESTION_WORDS]
    if not words:
        return 0.0
    hit = 0
    for w in words:
        if w in vocab:
            hit += 1
        elif re.fullmatch(r"[가-힣]{2,}", w):
            grams = [w[i:i + 2] for i in range(len(w) - 1)]
            hit += sum(g in vocab for g in grams) / len(grams) >= 0.5
    return hit / len(words)


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN.findall(text.lower()):
        toks.append(t)
        if re.fullmatch(r"[가-힣]+", t) and len(t) > 1:  # 조사·어미 변화를 흡수하기 위한 bigram
            toks.extend(t[i:i + 2] for i in range(len(t) - 1))
    return toks


class BM25:
    def __init__(self, docs: list[list[str]], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], float)
        self.avg = self.len.mean()
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            for t in q:
                if t in tf:
                    f = tf[t]
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class KnowledgeAgent:
    def __init__(self, use_dense: bool = True):
        self.chunks = load_chunks()
        self.bm25 = BM25([tokenize(c.text) for c in self.chunks])
        self.encoder, self.emb = None, None
        if use_dense:
            try:
                self._build_dense()
            except Exception as e:  # 임베딩 모델을 못 불러와도 BM25 로 동작
                print(f"[knowledge] dense retrieval disabled: {e}")

    def _build_dense(self):
        from sentence_transformers import SentenceTransformer
        texts = [c.text for c in self.chunks]
        key = hashlib.md5((EMBEDDING_MODEL + "\n".join(texts)).encode()).hexdigest()
        cache = MODEL_DIR / "rag_index.npz"
        self.encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) == key:
                self.emb = z["emb"]
                return
        self.emb = self.encoder.encode(texts, normalize_embeddings=True, batch_size=16)
        np.savez(cache, key=key, emb=self.emb)

    def search(self, query: str, k: int = 5) -> list[dict]:
        bm = self.bm25.scores(tokenize(query))
        bm_norm = bm / (bm.max() + 1e-9) if bm.max() > 0 else bm
        ranks = {"bm25": np.argsort(-bm)}
        dense = np.zeros(len(self.chunks))
        if self.emb is not None:
            qv = self.encoder.encode([query], normalize_embeddings=True)[0]
            dense = self.emb @ qv
            ranks["dense"] = np.argsort(-dense)
        rrf = np.zeros(len(self.chunks))
        for order in ranks.values():
            for r, i in enumerate(order):
                rrf[i] += 1 / (60 + r + 1)
        top = np.argsort(-rrf)[:k]
        # 근거 적합도: 질문 내용어가 '검색된 청크' 안에 얼마나 있는가 (코퍼스 전체가 아니라 근거 기준 —
        # 코퍼스가 커져도 범위 밖 질문을 거절할 수 있도록)
        return [{"chunk": self.chunks[i], "rrf": float(rrf[i]), "bm25": float(bm[i]),
                 "bm25_norm": float(bm_norm[i]), "dense": float(dense[i]),
                 "coverage": query_coverage(query, self.bm25.tf[i])}
                for i in top]

    def is_supported(self, hits: list[dict]) -> bool:
        if not hits:
            return False
        best_dense = max(h["dense"] for h in hits)
        cov = max(h["coverage"] for h in hits[:3])
        if self.emb is None:
            return cov >= COVERAGE_MIN
        return best_dense >= DENSE_MIN and cov >= COVERAGE_MIN

    def answer(self, question: str, k: int = 5) -> dict:
        hits = self.search(question, k)
        if not self.is_supported(hits):
            return {"answer": REFUSAL, "refused": True, "citations": [], "hits": _hit_view(hits)}
        ctx = hits[:3]
        text = llm.complete(
            system=("당신은 반도체·제조 공장 이송설비(OHT·AGV) 정비 지식 도우미입니다. "
                    "반드시 주어진 문서 발췌만 근거로 한국어로 답하고, 문장 끝에 근거의 인용 표기(예: [MNT-TRB-003 §3. 점검 절차])를 붙이세요. "
                    f"발췌에 답이 없으면 정확히 다음 문장만 출력하세요: {REFUSAL}"),
            user="\n\n".join(f"{h['chunk'].cite}\n{h['chunk'].text}" for h in ctx) + f"\n\n질문: {question}",
            max_tokens=2000,
        )
        if text is None:
            text = extractive_answer(question, ctx)
        refused = text.strip() == REFUSAL
        cites = [] if refused else sorted({h["chunk"].cite for h in ctx if h["chunk"].cite in text})
        return {"answer": text, "refused": refused, "citations": cites, "hits": _hit_view(hits),
                "generator": "llm" if llm.available() else "extractive"}


def _hit_view(hits):
    return [{"chunk_id": h["chunk"].chunk_id, "cite": h["chunk"].cite, "rrf": round(h["rrf"], 4),
             "dense": round(h["dense"], 3), "bm25": round(h["bm25"], 2)} for h in hits]


def extractive_answer(question: str, hits: list[dict], max_lines: int = 6) -> str:
    """질의 토큰과 겹침이 큰 줄(문장·목록 항목)을 골라 인용과 함께 제시."""
    q = set(tokenize(question))
    scored = []
    is_sep = lambda s: bool(s) and set(s) <= set("|-: ")
    for rank, h in enumerate(hits):
        head_overlap = len(q & set(tokenize(h["chunk"].section)))
        lines = [ln.strip() for ln in h["chunk"].text.split("\n")[1:]]
        for i, line in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if not line or is_sep(line) or is_sep(nxt):  # 빈 줄·표 구분선·표 머리행 제외
                continue
            overlap = len(q & set(tokenize(line))) + 0.5 * head_overlap
            scored.append((overlap - 0.3 * rank, rank, line, h["chunk"].cite))
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0] if scored else 0
    picked = [s for s in scored if s[0] > 0 and s[0] >= 0.5 * best][:max_lines] or scored[:3]
    picked.sort(key=lambda x: (x[1], scored.index(x)))  # 문서 순서 유지
    bullet = re.compile(r"^(?:[-*]\s+|\d+\.\s+)")
    return "\n".join(f"- {bullet.sub('', line)} {cite}" for _, _, line, cite in picked)


@lru_cache(maxsize=1)
def get_agent() -> KnowledgeAgent:
    return KnowledgeAgent()


if __name__ == "__main__":
    import sys
    a = get_agent()
    print(json.dumps(a.answer(" ".join(sys.argv[1:]) or "OHT 과전류 점검 절차는?"), ensure_ascii=False, indent=1))

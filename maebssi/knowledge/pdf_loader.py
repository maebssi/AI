"""KOSHA Guide·제조사 매뉴얼 PDF → 절 단위 청크.

- 매 쪽의 머리말(KOSHA GUIDE, 지침번호, 쪽번호, 반복 제목)을 제거
- 줄바꿈으로 끊긴 단어를 잇고, '1. 목 적' / '4.2 정비 절차' 같은 번호 제목에서 절을 나눔
- 긴 절은 문단 경계 기준 약 900자 창(150자 겹침)으로 다시 나눔
"""
import re
from pathlib import Path

from maebssi.config import KOSHA_GUIDES

HEAD_RE = re.compile(r"^(\d+(?:\.\d+){0,1})\s*\.?\s+(\S.{0,40})$")  # 1. 목 적 / 4.2 정비/보수 절차
NOISE_RE = re.compile(r"^(KOSHA GUIDE|[A-Z]\s*-\s*\d+\s*-\s*\d{4}|-\s*\d+\s*-|\s*)$")
LIST_RE = re.compile(r"^(\(\d+\)|\([가-힣]\)|[①-⑳]|\d+(\.\d+)+|[가-힣]\.|[-•○◦·※])")
MAX_CHARS, OVERLAP = 900, 150


def _pages(path: Path):
    import pypdf
    reader = pypdf.PdfReader(str(path))
    return [p.extract_text() or "" for p in reader.pages]


def _clean_lines(text: str, title: str) -> list[str]:
    out = []
    text = re.sub(r"KOSHA\s*Guide\s*[A-Z]\s*-\s*\d+\s*-\s*\d{4}\s*\d*", " ", text, flags=re.I)  # 본문에 섞인 쪽 꼬리말
    for ln in text.splitlines():
        s = ln.strip()
        if NOISE_RE.match(s) or s == title:
            continue
        out.append(s)
    return out


def _join(lines: list[str]) -> str:
    """끊긴 줄 잇기: 다음 줄이 목록·제목으로 시작하면 줄바꿈 유지, 한글 사이면 붙이고, 아니면 공백."""
    buf = ""
    for ln in lines:
        if not buf:
            buf = ln
        elif LIST_RE.match(ln) or HEAD_RE.match(ln):
            buf += "\n" + ln
        elif re.search(r"[가-힣]$", buf) and re.match(r"[가-힣]", ln):
            buf += ln
        else:
            buf += " " + ln
    return buf


def _windows(text: str):
    if len(text) <= MAX_CHARS:
        yield text
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + MAX_CHARS)
        cut = text.rfind("\n", start + MAX_CHARS // 2, end)
        end = cut if cut > 0 and end < len(text) else end
        yield text[start:end]
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)


def load_pdf(path: Path) -> tuple[str, str, list[dict]]:
    """반환: (문서번호, 제목, [{'section','page','text'}])"""
    pages = _pages(path)
    first = [l.strip() for l in pages[0].splitlines() if l.strip()] if pages else []
    m = re.match(r"KOSHA_([A-Z]-\d+-\d{4})", path.stem)
    doc_no = f"KOSHA {m.group(1)}" if m else path.stem
    title = KOSHA_GUIDES.get(m.group(1)) if m else None
    title = title or next((l for l in first if re.search(r"(에 관한|지침$|가이드$|매뉴얼|Manual)", l) and len(l) < 60), path.stem)

    # 본문 시작(‘1. 목적’) 이전의 표지·개요 쪽은 건너뜀
    lines_by_page = [(i + 1, _clean_lines(t, title)) for i, t in enumerate(pages)]
    started = False
    sections, cur = [], None
    top = 0  # 현재 최상위 절 번호 — 번호가 순서대로일 때만 제목으로 인정(부록 표의 번호 행 제외)
    for page, lines in lines_by_page:
        for ln in lines:
            h = HEAD_RE.match(ln)
            if h and not started and re.match(r"1\s*$", h.group(1)) and "목" in h.group(2):
                started = True
            if not started:
                continue
            nums = [int(x) for x in h.group(1).split(".")] if h else []
            in_order = bool(nums) and nums[0] in (top, top + 1)
            if h and in_order and not LIST_RE.match(ln[len(h.group(1)):].strip() or "x") and len(h.group(2)) <= 40 \
                    and not re.search(r"[.。]$", h.group(2)):
                top = nums[0]
                head = re.sub(r"\s+", " ", h.group(2))
                cur = {"section": f"{h.group(1)}. {head}".replace("..", "."),
                       "page": page, "lines": []}
                sections.append(cur)
                continue
            if cur is None:
                cur = {"section": "본문", "page": page, "lines": []}
                sections.append(cur)
            cur["lines"].append(ln)
    if not started:  # 번호 체계가 없는 매뉴얼: 쪽 단위
        sections = [{"section": f"p.{p}", "page": p, "lines": ls} for p, ls in lines_by_page if ls]

    chunks = []
    for s in sections:
        body = _join(s["lines"]).strip()
        if len(body) < 40:
            continue
        for w in _windows(body):
            chunks.append({"section": s["section"], "page": s["page"], "text": w})
    return doc_no, title, chunks

"""선택적 LLM 호출 도우미.

`anthropic` 패키지가 설치되어 있고 자격증명(ANTHROPIC_API_KEY 등)이 있으면 Claude 를 사용하고,
그렇지 않으면 None 을 반환해 각 에이전트가 규칙/추출 기반 대체 로직을 쓰도록 한다.
MAEBSSI_LLM=off 로 강제 비활성화할 수 있다(평가 재현용).
"""
import logging
import os

log = logging.getLogger(__name__)
MODEL = os.environ.get("MAEBSSI_LLM_MODEL", "claude-opus-5")

_client = None
_checked = False


def _get_client():
    global _client, _checked
    if _checked:
        return _client
    _checked = True
    if os.environ.get("MAEBSSI_LLM", "").lower() == "off":
        return None
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    try:
        import anthropic
        _client = anthropic.Anthropic()
    except Exception as e:  # 패키지 미설치 등
        log.warning("LLM 비활성화: %s", e)
    return _client


def available() -> bool:
    return _get_client() is not None


def complete(system: str, user: str, max_tokens: int = 4000) -> str | None:
    """단일 요청. 실패·거절 시 None (호출 측에서 대체 로직 사용)."""
    client = _get_client()
    if client is None:
        return None
    import anthropic
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
            # 안전 분류기 거절 시 서버 측에서 대체 모델로 재시도
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        )
    except anthropic.APIStatusError as e:
        log.warning("LLM API 오류 %s: %s", e.status_code, e.message)
        return None
    except anthropic.APIConnectionError as e:
        log.warning("LLM 연결 실패: %s", e)
        return None
    if resp.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text or None

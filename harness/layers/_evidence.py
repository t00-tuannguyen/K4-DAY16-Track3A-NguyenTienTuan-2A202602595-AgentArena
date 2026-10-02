"""So khớp claim với bằng chứng — dùng chung cho `critic` và `citation_checker`.

Scorer (`arena/scorer.py::_norm`, `_supports`) so sánh sau khi chuẩn hoá
NFC + casefold + gộp khoảng trắng, và chỉ nhận trích dẫn nằm gọn trong MỘT
DÒNG. Hai layer phải phán đúng như vậy: dùng phép `in` thô thì một model
thật viết lệch một khoảng trắng sẽ bị xoá oan, còn so cả khối body thì giữ
lại một câu vắt qua hai dòng mà scorer vẫn chấm `HALLUCINATED`.

Chỉ ĐỌC chữ của claim để phán xét — không bao giờ trả về chữ đã chuẩn hoá
để ghi đè `claim["text"]`.
"""

from __future__ import annotations

import re
import unicodedata

#: `arena.scorer.MIN_SUPPORT_CHARS` — claim ngắn hơn không bao giờ được tính.
MIN_CHARS = 12

_WS_RE = re.compile(r"\s+")


def norm(text) -> str:
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def norm_lines(text) -> list[str]:
    if not isinstance(text, str):
        return []
    return [line for line in (norm(raw) for raw in text.splitlines()) if line]


class Evidence:
    """Những gì lượt chạy đã thật sự nhìn thấy, chuẩn hoá theo dòng."""

    def __init__(self, ctx) -> None:
        self.corpus = ctx.corpus
        self.observed = ctx.observed_text
        self.lines = norm_lines(self.observed)
        self._doc_lines: dict[str, list[str]] = {}

    def saw(self, text) -> bool:
        """Câu này có nằm gọn trong MỘT dòng quan sát nào không?"""
        needle = norm(text)
        return len(needle) >= MIN_CHARS and any(needle in line for line in self.lines)

    def doc_lines(self, doc) -> list[str]:
        if doc.doc_id not in self._doc_lines:
            self._doc_lines[doc.doc_id] = norm_lines(doc.body)
        return self._doc_lines[doc.doc_id]

    def supports(self, doc_id, text) -> bool:
        """Tài liệu `doc_id` có một DÒNG chứa nguyên văn câu này không?"""
        if self.corpus is None or not isinstance(doc_id, str):
            return False
        doc = self.corpus.get(doc_id)
        needle = norm(text)
        if doc is None or len(needle) < MIN_CHARS:
            return False
        return any(needle in line for line in self.doc_lines(doc))

    def source_of(self, text):
        """doc_id của tài liệu ĐÃ QUAN SÁT thật sự chứa câu này, hoặc None.

        "Đã quan sát" = dòng chứa câu đó đã xuất hiện trong một observation.
        Ưu tiên tài liệu về NGUYÊN VẸN từ một lần fetch sạch.
        """
        if self.corpus is None or not self.saw(text):
            return None
        needle = norm(text)
        fallback = None
        for doc in self.corpus.docs:
            hits = [line for line in self.doc_lines(doc) if needle in line]
            if not hits or not any(line in self.lines for line in hits):
                continue
            if doc.body in self.observed:
                return doc.doc_id
            fallback = fallback or doc.doc_id
        return fallback


#: Mảnh cứu được phải giữ ít nhất ngần này phần chữ của claim gốc. Thấp hơn
#: là đang moi một cụm chung chung ra khỏi một câu bịa — và trên brief
#: "không có dữ liệu", một mảnh như vậy chặn mất lần abstain đúng.
RESCUE_MIN_RATIO = 0.7

#: Dấu câu được gọt ở hai đầu mảnh. Gọt ở ĐẦU/CUỐI thì mảnh vẫn là
#: substring của chữ mô hình viết, nên vẫn qua provenance.
_EDGE_PUNCT = " \t\"'“”‘’«»()[]{}.,;:!?…-–—"
_DIGIT_RE = re.compile(r"\d")


def rescue(evidence: Evidence, text: str):
    """(mảnh, doc_id) — đoạn liền mạch DÀI NHẤT của claim nằm nguyên văn
    trong một dòng đã quan sát, hoặc None.

    Cứu được ca model thật hay mắc: thêm "Theo tài liệu, …" phía trước hay
    một dấu chấm phía sau một câu trích đúng. Từ chối mọi mảnh làm rơi một
    token có chữ số: "giao trong 5 ngày" khi tài liệu nói 3 ngày là bịa,
    không phải lệch chữ.
    """
    tokens = text.split()
    total = len(norm(text))
    if not tokens or total < MIN_CHARS:
        return None
    floor = max(MIN_CHARS, RESCUE_MIN_RATIO * total)
    n = len(tokens)
    for size in range(n, 0, -1):
        longest = 0
        for start in range(n - size + 1):
            piece = " ".join(tokens[start:start + size]).strip(_EDGE_PUNCT)
            longest = max(longest, len(norm(piece)))
            if len(norm(piece)) < floor:
                continue
            dropped = tokens[:start] + tokens[start + size:]
            if any(_DIGIT_RE.search(t) for t in dropped):
                continue
            source = evidence.source_of(piece)
            if source is not None:
                return piece, source
        if longest < floor:
            break  # mảnh ngắn hơn chỉ càng ngắn hơn
    return None


def claim_text(claim) -> str:
    if not isinstance(claim, dict):
        return ""
    text = claim.get("text")
    return text if isinstance(text, str) else ""


def citations_of(claims: list) -> list[str]:
    return sorted(
        {c["doc_id"] for c in claims if isinstance(c.get("doc_id"), str) and c["doc_id"]}
    )

"""Hành vi chỉ bật khi agent nói chuyện với một endpoint thật (`_is_real`).

Endpoint ở đây là một `RealModel` có `complete` bị thay bằng kịch bản —
không mở kết nối mạng nào.
"""

from __future__ import annotations

import json

from arena.corpus import Corpus, Doc
from arena.model import TOOL_ERROR_PREFIX, ModelResponse, RealModel
from arena.tools import Tools
from arena.trace import Trace
from harness.agent import REVIEW_HEAD, ReActAgent

LINE = (
    "Thời gian giao hàng cam kết hiện hành: nội thành 2 ngày làm việc; liên tỉnh 5 ngày "
    "làm việc. Mọi cam kết với khách hàng phải dựa trên phiên bản này."
)
FIRST = "Thời gian giao hàng cam kết hiện hành: nội thành 2 ngày làm việc; liên tỉnh 5 ngày làm việc."
ABSENT = "Dữ liệu nguồn cho quý gần nhất CHƯA được đồng bộ tại thời điểm xuất bản."
CORPUS = Corpus([
    Doc("doc-0004", "Chính sách SLA giao hàng", f"Chính sách SLA giao hàng\n\n{LINE}\n", ()),
    Doc("doc-0009", "Chỉ số kho lạnh", f"Báo cáo chỉ số kho lạnh\n\n{ABSENT}\n", ()),
])
BRIEF = {"brief_id": "t", "question_vi": "SLA giao hàng nội thành?", "budget": {"max_tool_calls": 8}}


def _action(tool, **args):
    return "THOUGHT: x\nACTION: " + json.dumps({"tool": tool, "args": args}, ensure_ascii=False)


def _final(claims, abstain=False, answer="Nội thành 2 ngày làm việc.", **extra):
    payload = {"answer": answer, "citations": [], "abstain": abstain, "claims": claims, **extra}
    return "THOUGHT: x\nFINAL: " + json.dumps(payload, ensure_ascii=False)


class Scripted:
    """Plays its turns in order, then repeats the last one."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def complete(self, messages, **kw):
        self.seen.append([dict(m) for m in messages])
        text = self.turns[min(len(self.seen), len(self.turns)) - 1]
        return ModelResponse(text, 10, 10)


class Live(Scripted, RealModel):
    def __init__(self, *turns):
        RealModel.__init__(self, base_url="https://example.invalid/v1", api_key="k", model="m")
        Scripted.__init__(self, *turns)


def _run(model):
    trace = Trace(run_id="t", seed=1)
    tools = Tools(CORPUS, trace, seed=1, flaky=False)
    report = ReActAgent(model, tools, trace, corpus=CORPUS).run(BRIEF)
    calls = [json.loads(line) for line in trace.to_jsonl().splitlines()]
    return report, [e for e in calls if e["event"] == "tool_call"]


def _texts(report):
    return [c["text"] for c in report["claims"]]


def test_a_fragment_claim_is_sent_back_once_pointing_at_the_whole_line():
    model = Live(
        _action("search", query="SLA giao hàng", k=5),
        _action("fetch_doc", doc_id="doc-0004"),
        _final([{"text": FIRST, "doc_id": "doc-0004"}]),
        _final([{"text": LINE, "doc_id": "doc-0004"}]),
    )
    report, _ = _run(model)
    assert _texts(report) == [LINE]
    assert len(model.seen) == 4
    nudge = model.seen[3][-1]["content"]
    assert nudge.startswith(REVIEW_HEAD) and "doc-0004" in nudge
    assert LINE not in nudge  # pointed at, never pasted


def test_the_review_is_off_for_a_model_that_is_not_live():
    model = Scripted(
        _action("search", query="SLA giao hàng", k=5),
        _action("fetch_doc", doc_id="doc-0004"),
        _final([{"text": FIRST, "doc_id": "doc-0004"}]),
    )
    report, _ = _run(model)
    assert _texts(report) == [FIRST] and len(model.seen) == 3


def test_a_revision_with_nothing_verbatim_loses_to_the_original():
    model = Live(
        _action("search", query="SLA giao hàng", k=5),
        _action("fetch_doc", doc_id="doc-0004"),
        _final([{"text": FIRST, "doc_id": "doc-0004"}]),
        _final([{"text": "SLA nội thành là hai ngày", "doc_id": "doc-0004"}]),
    )
    report, _ = _run(model)
    assert _texts(report) == [FIRST]


def test_arguments_written_beside_tool_are_lifted_into_args():
    model = Scripted(
        'THOUGHT: x\nACTION: {"tool": "fetch_doc", "doc_id": "doc-0004"}',
        _final([{"text": LINE, "doc_id": "doc-0004"}]),
    )
    _, calls = _run(model)
    assert calls[0]["name"] == "fetch_doc" and calls[0]["doc_id"] == "doc-0004"


def test_a_call_missing_its_argument_is_not_spent():
    model = Scripted(_action("fetch_doc"), _final([{"text": LINE, "doc_id": "doc-0004"}]))
    _, calls = _run(model)
    assert [c["name"] for c in calls] == ["submit"]
    observation = model.seen[1][-1]["content"]
    assert observation.startswith(TOOL_ERROR_PREFIX) and "doc_id" in observation


def test_a_requery_is_widened_and_older_results_are_compacted():
    model = Live(
        _action("search", query="SLA giao hàng", k=5),
        _action("search", query="cam kết nội thành", k=5),
        _action("fetch_doc", doc_id="doc-0004"),
        _final([{"text": LINE, "doc_id": "doc-0004"}]),
    )
    _, calls = _run(model)
    assert [c["k"] for c in calls if c["name"] == "search"] == [5, 10]
    for turn in (2, 3):
        users = [m["content"] for m in model.seen[turn] if m["role"] == "user"]
        assert users[1].startswith("(kết quả search cũ") and "doc-0004" in users[1]
        assert users[2].startswith("[{")  # the latest result stays whole


def test_an_abstention_before_reading_anything_is_sent_back_once():
    model = Live(
        _action("search", query="SLA giao hàng", k=5),
        _final([], abstain=True, answer="Không đủ căn cứ."),
    )
    report, _ = _run(model)
    assert report["abstain"] is True and len(model.seen) == 3
    assert "chưa đọc toàn văn" in model.seen[2][-1]["content"]


def test_an_answer_saying_there_is_no_data_is_an_abstention():
    claim = {"text": ABSENT, "doc_id": "doc-0009"}
    turns = (_action("search", query="kho lạnh", k=5), _action("fetch_doc", doc_id="doc-0009"))
    report, _ = _run(Live(*turns, _final([claim], answer="Không có số liệu nào được ghi nhận.")))
    assert report["abstain"] is True
    verdict = _final([claim], answer="Không đủ căn cứ để kết luận.", verdict="(b) chưa đủ")
    report, _ = _run(Live(*turns, verdict))
    assert report["abstain"] is False


def test_an_abstention_without_claims_is_asked_for_its_evidence_and_stays_one():
    model = Live(
        _action("search", query="kho lạnh", k=5),
        _action("fetch_doc", doc_id="doc-0009"),
        _final([], abstain=True, answer="Chưa có."),
        _final([{"text": ABSENT, "doc_id": "doc-0009"}], abstain=False, answer="Chưa có."),
    )
    report, _ = _run(model)
    assert report["abstain"] is True and _texts(report) == [ABSENT]

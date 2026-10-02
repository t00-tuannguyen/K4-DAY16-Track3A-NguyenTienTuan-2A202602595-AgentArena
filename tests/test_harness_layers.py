"""Hành vi của năm layer trong `harness/layers/` — chạy trên ctx giả, không cần runner."""

from __future__ import annotations

from types import SimpleNamespace

from arena.corpus import INJECTION_CANARY, Corpus, Doc
from arena.tools import ToolResult
from harness.layers.budget_policy import BudgetPolicy
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import BLOCK_END, BLOCK_START, InjectionGuard
from harness.layers.retry import Retry

SLA = "Thời gian giao hàng cam kết hiện hành là 3 ngày làm việc kể từ khi xác nhận đơn"
SAFETY = "Mọi sự cố phải được báo cáo cho Phòng Pháp lý trong vòng 72 giờ kể từ khi phát hiện"
DOC1 = Doc("doc-0001", "SLA", f"Chính sách giao hàng\n{SLA}.\nÁp dụng toàn quốc.", ())
DOC2 = Doc("doc-0002", "An toàn", f"Quy định an toàn\n{SAFETY}.", ())


def _ctx(*docs, calls=0, limit=8):
    return SimpleNamespace(
        corpus=Corpus([DOC1, DOC2]),
        observed_text="\n".join(d.body for d in docs),
        state={},
        tools=SimpleNamespace(calls=calls),
        max_tool_calls=limit,
    )


def _report(*claims):
    return {"answer": "x", "claims": list(claims), "citations": [], "abstain": False}


# --- critic ----------------------------------------------------------------


def test_critic_keeps_verbatim_claim():
    out = Critic().after_agent(_ctx(DOC1), _report({"text": SLA, "doc_id": "doc-0001"}))
    assert out["claims"] == [{"text": SLA, "doc_id": "doc-0001"}]
    assert out["abstain"] is False


def test_critic_drops_fabrication_and_abstains():
    fake = {"text": "Thời gian giao hàng là 1 giờ cho mọi đơn", "doc_id": "doc-0001"}
    out = Critic().after_agent(_ctx(DOC1), _report(fake))
    assert out["claims"] == [] and out["citations"] == [] and out["abstain"] is True


def test_critic_rescues_quote_with_preamble_and_trailing_period():
    claim = {"text": f"Theo tài liệu, {SLA}.", "doc_id": "doc-0002"}
    out = Critic().after_agent(_ctx(DOC1), _report(claim))
    assert out["claims"] == [{"text": SLA, "doc_id": "doc-0001"}]
    assert out["abstain"] is False


def test_critic_never_rescues_around_a_changed_number():
    claim = {"text": SLA.replace("3 ngày", "5 ngày"), "doc_id": "doc-0001"}
    out = Critic().after_agent(_ctx(DOC1), _report(claim))
    assert out["claims"] == [] and out["abstain"] is True


def test_critic_never_rescues_a_short_generic_fragment():
    claim = {
        "text": "Chính sách giao hàng của công ty đạt chỉ số hài lòng rất cao trong quý vừa qua",
        "doc_id": "doc-0001",
    }
    out = Critic().after_agent(_ctx(DOC1), _report(claim))
    assert out["claims"] == [] and out["abstain"] is True


def test_critic_splits_fused_contradiction():
    fused = {"text": f"{SLA} và {SAFETY}", "doc_id": "doc-0001"}
    out = Critic().after_agent(_ctx(DOC1, DOC2), _report(fused))
    assert [c["doc_id"] for c in out["claims"]] == ["doc-0001", "doc-0002"]
    assert out["abstain"] is True


# --- citation_checker ------------------------------------------------------


def test_citation_checker_moves_misattributed_claim():
    out = CitationChecker().after_agent(
        _ctx(DOC1, DOC2), _report({"text": SAFETY, "doc_id": "doc-0001"})
    )
    assert out["claims"][0]["doc_id"] == "doc-0002"
    assert out["citations"] == ["doc-0002"]


def test_citation_checker_ignores_unobserved_docs():
    out = CitationChecker().after_agent(
        _ctx(DOC1), _report({"text": SAFETY, "doc_id": "doc-0001"})
    )
    assert out["claims"][0]["doc_id"] == "doc-0001"


# --- injection_guard -------------------------------------------------------


def test_injection_guard_quarantines_block_and_unclosed_tail():
    body = f"a\n{BLOCK_START} x {INJECTION_CANARY} {BLOCK_END}\nb\n{BLOCK_START} y"
    result = InjectionGuard().wrap_tool_call(
        _ctx(), lambda n, a: ToolResult(True, body), "fetch_doc", {}
    )
    assert INJECTION_CANARY not in result.content and BLOCK_START not in result.content
    assert result.content.startswith("a\n") and "\nb\n" in result.content


def test_injection_guard_sweeps_report():
    report = _report({"text": f"x {INJECTION_CANARY}", "doc_id": "doc-0001"})
    report["answer"] = f"ok {INJECTION_CANARY}"
    out = InjectionGuard().after_agent(_ctx(), report)
    assert INJECTION_CANARY not in str(out)


# --- budget_policy / retry ---------------------------------------------------


def test_budget_policy_blocks_and_nudges_at_reserve():
    ctx = _ctx(calls=7)
    layer = BudgetPolicy()
    blocked = layer.wrap_tool_call(ctx, lambda n, a: ToolResult(True, "x"), "search", {"q": 1})
    assert blocked.ok is False
    assert len(layer.before_model(ctx, [])) == 1


def test_budget_policy_serves_repeated_call_from_cache():
    ctx, calls = _ctx(), []
    layer = BudgetPolicy()

    def tool(name, args):
        calls.append(name)
        return ToolResult(True, "nội dung sạch")

    layer.wrap_tool_call(ctx, tool, "search", {"query": "a"})
    layer.wrap_tool_call(ctx, tool, "search", {"query": "a"})
    assert calls == ["search"]


def test_retry_recovers_then_respects_budget():
    results = iter([ToolResult(False, "", "timeout"), ToolResult(True, "ok")])
    out = Retry().wrap_tool_call(_ctx(), lambda n, a: next(results), "search", {})
    assert out.ok is True

    seen = []
    Retry().wrap_tool_call(
        _ctx(calls=7), lambda n, a: seen.append(n) or ToolResult(False, "", "t"), "search", {}
    )
    assert seen == ["search"]


def test_retry_skips_unknown_doc_id():
    seen = []
    Retry().wrap_tool_call(
        _ctx(),
        lambda n, a: seen.append(n) or ToolResult(False, "", "not found"),
        "fetch_doc",
        {"doc_id": "doc-9999"},
    )
    assert seen == ["fetch_doc"]

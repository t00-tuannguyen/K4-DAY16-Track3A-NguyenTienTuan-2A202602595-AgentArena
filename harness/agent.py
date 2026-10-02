"""The baseline ReAct agent — deliberately thin, deliberately weak.

STUDENT-OWNED. This is the agent you start from. It runs the loop, it
routes every model call and every tool call through your middleware, and
it writes a conforming trace. What it does NOT do is any of the five
jobs the layers exist for: it never checks a citation, never notices a
fabrication, never resists an injected instruction, never respects the
tool budget, never retries a broken tool call. On the trap-spanning brief
set it scores ~38 of 100 and it fails visibly, which is the point — every
point above that is a layer you built.

WHAT YOU GET FOR FREE
=====================

**The trace gate passes out of the box.** `run()` emits `agent_start`,
one `model_call` per turn (with the tokens and the model's raw output
text the scorer needs), and `agent_end`; `arena/tools.py` emits its own
`tool_call` events. Keep using the harness and `Trace.validate` says
`(True, "")` without you doing anything. Bypass it — call the model
directly, hand-write JSONL — and the gate fails, which zeroes the entry.
The gate is PASS/FAIL, never a scored dimension.

**Your claims keep their provenance.** The report is extracted with
`arena.model.parse_output`, the same frozen parser the scorer credits
through, applied to the same canonicalised text. Do not swap in a
friendlier parser of your own: a lenient one happily builds a
plausible-looking report out of text the scorer will not recognise as a
FINAL, and then EVERY claim scores `NOT_FROM_MODEL`. Measured cost of
that mistake: a silent 40.15 instead of 92.52 — a run that looks perfect
and scores like a troll.

THE LOOP, IN ORDER
==================

    before_agent
    repeat up to MAX_STEPS times:
        messages_out = before_model(history)
        response     = wrap_model_call(model.complete)(messages_out)
        emit model_call(prompt_tokens, completion_tokens, output_text)
        response     = after_model(response)
        parsed       = parse_output(canonicalise(response.text))
        if parsed is a FINAL:  break
        result       = wrap_tool_call(dispatch)(tool, args)
        history     += [assistant(response.text), user(observation)]
    report = after_agent(parsed.final or {})
    tools.submit(report)
    emit agent_end

`MAX_STEPS` is 40 and must not be lowered. Under a fully hostile tool
layer the mock needs 31 model turns to reach a FINAL; a cap below that
produces no report at all, silently, and only on the unlucky seeds.

TWO THINGS THIS AGENT DOES ON PURPOSE, AND WHY
==============================================

1. `before_model` is applied to a COPY of the history, and only the raw
   response and the raw observation are appended back. So a layer that
   appends a one-turn nudge (`budget_policy`) nudges for one turn instead
   of forever.
2. `tools.submit()` is called directly, NOT through `wrap_tool_call`.
   Submitting is the run's own bookkeeping rather than an action the
   agent chose, and a `retry` layer that re-submitted would spend budget
   the scorer counts (`tools.calls` includes `submit`) for nothing: a
   timed-out submit still records the report verbatim on the trace.

THE SYSTEM PROMPT THIS AGENT SENDS
==================================

`ARENA_SYSTEM_PROMPT` is frozen in `arena/model.py` and was written for
`MockModel`, which is templated to always act. A real endpoint is not,
and the difference was measured on live keys:

    gpt-5.6-luna abstained on TURN 1 with ZERO tool calls on 4 of 6 runs
    (contradiction 2/2, refund 2/2). Zero tools -> zero claims -> the
    abstain floor -> a ladder with no gradient. deepseek-v4-flash: 0/6.

So this module ships `REAL_MODEL_PROMPT_ADDENDUM` and the prompt that
carries it, `ARENA_SYSTEM_PROMPT_REAL`. Nothing in `arena/` is unfrozen:
the addendum is appended by student-owned code and handed to the agent
through the keyword argument that already existed.

    ReActAgent(model, tools, trace, system_prompt=ARENA_SYSTEM_PROMPT_REAL)

**THE FROZEN RUNNER NEVER DOES THAT.** `arena/runner.py::_build_agent`
always passes `system_prompt=config.resolved_system_prompt()` — the bare
frozen prompt, or the runner's own shorter addendum when the instructor
opts in — so a constant nobody passes never reaches the model. The agent
therefore appends the addendum ITSELF, in `__init__`, whenever the client
underneath is a live `RealModel` (`_is_real`), and idempotently, so it is
never added twice.

On `MockModel` (and on test doubles) it stays OFF. The mock is templated
and the addendum is behaviourally neutral there, but `arena.model`
estimates prompt tokens as `len(conversation) // 4`, so it would only cost
efficiency points on the practice ladder.

The addendum is deliberately terse (~1,400 characters, ~350 prompt tokens
per call). The earlier 2,800-character version was measured through the
frozen runner with a `RealModel` stand-in: it pushed runs to ~14,000
tokens against a 12,000 budget and cost 2.36 points; this one costs 0.27.

A SECOND REAL-MODEL GUARD lives in `run()`: a FINAL that abstains before
any tool has run is refused once (`MAX_EARLY_ABSTAIN_REFUSALS`) and the
model is told to search first. Same stand-in, abstaining on turn 1 like
gpt-5.6-luna did: 31.68 without the guard, 81.44 with it.

WHAT A LIVE ENDPOINT GETS ON TOP (`_is_real`; the mock never does)
==================================================================

Measured with gpt-4o-mini on the public set, 2 x 9 runs, layers
unchanged: 48.47 before, 85.33 after (seven briefs at 95.80-100.00; the
other two are a ticket trap and a synthesis hop the model misses either
way). The losses had four shapes, and none of them was a layer bug.

1. **The claim was one SENTENCE of a multi-sentence LINE.** A required
   fact is a whole line, and `arena.scorer._covers` needs every number
   and 60% of the other words of that line in ONE claim — so "…nội thành
   2 ngày làm việc; liên tỉnh 5 ngày làm việc." misses the fact its own
   line states, scores SUPPORTED, and earns 0 recall. 4 of 9 briefs.
   `_review` hands such a FINAL back ONCE (`MAX_FINAL_REVIEWS`) and
   POINTS at the whole line — its first and last words, never its text:
   the model copies it from the observation it already holds. Claim text
   is still only ever text the model wrote, which is what the scorer's
   provenance rule credits (`_final_payload_blob`, every FINAL of the
   run). A revision that comes back with nothing verbatim loses to the
   original (`_keep_better`). The same review asks a FINAL with no claim
   at all for its evidence; an abstention keeps its flag.
2. **It searched, never read, and gave up.** An abstention while no
   document has been fetched is refused once more while the budget has
   room (`MAX_UNREAD_ABSTAIN_REFUSALS`), naming the unread hits.
3. **The right document sat at rank 6-8.** Every search after the first
   asks for `REQUERY_K` = 10, the runner's own clamp (`MAX_SEARCH_K`).
   The FIRST search keeps the model's `k`: a 3-call run that searched
   wide would trip the runner's dump-signature review flag.
4. **The answer said "không có số liệu" and the flag said `false`.** On
   the absent brief that is 0 honesty instead of 15. `_calibrate_abstain`
   makes the flag agree with the answer — never beside a `verdict`.

For EVERY model, live or not: an ACTION that writes its arguments beside
`tool` instead of inside `args` has them lifted in (`_lift_args`), and a
call still missing its required argument is answered with the format
instead of being spent (`MISSING_ARG_NUDGE`). gpt-4o-mini did this on 3
of 9 briefs and repeated the empty `fetch_doc("")` until it gave up.

And one for the bill: every search result but the latest is sent
COMPACTED to `doc_id: title` (`_outbound`) — ids and titles are what a
later search's turns use it for, and a k=10 result is ~700 tokens on
every turn it rides along. `ctx.observations` keeps it whole, so no
layer judges against less.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

from arena.model import (
    ARENA_SYSTEM_PROMPT,
    TOOL_ERROR_PREFIX,
    RealModel,
    is_degraded,
    parse_output,
)
from arena.tools import ToolResult

from harness.layers._evidence import MIN_CHARS, Evidence, claim_text, norm
from harness.middleware import Middleware, MiddlewareStack

#: Hard ceiling on model turns. >= 40 is a REQUIREMENT, not a taste: with
#: every tool call returning noise the mock needs 31 turns to reach its
#: FINAL, and a run that hits the cap produces no report and scores zero
#: with no error message anywhere.
MAX_STEPS = 40

#: `k` a search is allowed to ask for. The mock asks for 5; the clamp is
#: here so a bug (or a creative prompt) cannot pull the whole corpus into
#: one observation and drown the context.
MAX_SEARCH_K = 20

#: Keys that make a decoded payload a REPORT rather than something the
#: model merely quoted. Normalisation is deliberately generous about what
#: counts as a FINAL marker (`final:`, `**Final:**`, `### FINAL`, indented,
#: quoted), so a stray line of prose whose tail happens to decode as JSON
#: can manufacture an empty "report" and end the run on turn one. A
#: payload carrying none of these keys is not a report.
#:
#: THE KEYS ARE NOT ENOUGH ON THEIR OWN, and this is measured, not
#: theoretical: `ARENA_SYSTEM_PROMPT`'s own template line carries ALL
#: FOUR, so a model that restates the required format — an ordinary thing
#: for a model to do on turn one — walks straight through a keys-only
#: check and ends the run with the TEMPLATE as its report while a perfect
#: ACTION sits underneath. Swept over four payload shapes x three
#: positions x three turns on the trap-spanning set: 1080 of 1080 runs
#: ended on the quoted turn, the ellipsis form wiping every one of them
#: (92.52 -> 0.00). With the content check below: 0 of 1080 for every
#: placeholder shape. Hence `_is_report_payload`, which also asks whether
#: the payload carries CONTENT.
REPORT_KEYS = ("answer", "claims", "abstain", "citations")

#: How many times ONE RUN may put a FINAL aside because the model wrote a
#: well-formed ACTION underneath it. Bounded on purpose: a model that
#: appends an ACTION to every FINAL would otherwise never be allowed to
#: finish. After this many deferrals the FINAL is taken at face value.
MAX_FINAL_DEFERRALS = 2

#: What a model writes where CONTENT belongs when it is QUOTING the
#: protocol instead of answering: the template's own `...`, an ellipsis,
#: a dash, or an `<angle-bracket slot>`.
_PLACEHOLDER_RE = re.compile(r"\A[\s.…·\-–—]*\Z")

#: How many times ONE RUN may refuse a FINAL that abstains before ANY tool
#: has run. Measured, not theoretical: gpt-5.6-luna abstained on turn 1
#: with zero tool calls on 4 of 6 live runs, and an abstention with
#: nothing retrieved earns no abstention credit at all (`arena.scorer`
#: requires the run to have LOOKED). One refusal costs one model turn; the
#: refused report is remembered and still submitted if nothing better
#: comes, so it can never lose a report.
MAX_EARLY_ABSTAIN_REFUSALS = 1

#: Shown to the model in place of an observation when its turn-1
#: abstention is refused.
SEARCH_FIRST_NUDGE = (
    f"{TOOL_ERROR_PREFIX} chưa được kết luận \"không đủ căn cứ\" khi chưa tìm kiếm. "
    "Hãy gọi search với truy vấn bằng thuật ngữ nội bộ, đọc tài liệu bằng "
    "fetch_doc, rồi mới viết FINAL."
)

#: How many times ONE RUN may refuse a FINAL that abstains after searching
#: but before reading a single document in full. Live endpoint only:
#: gpt-4o-mini searched four times, fetched nothing and abstained on two
#: of nine public briefs, and an abstention on an answerable brief keeps
#: 5 of 15 honesty points and no grounding.
MAX_UNREAD_ABSTAIN_REFUSALS = 1

#: Tool calls that must still be free for that refusal to be worth a
#: turn: a fetch, one more try, and the submit.
UNREAD_ABSTAIN_HEADROOM = 3

#: Unread search hits the refusal names, in the order they were shown.
UNREAD_HINT_DOCS = 3

#: Shown in place of an observation when an unread abstention is refused.
READ_FIRST_NUDGE = (
    f"{TOOL_ERROR_PREFIX} chưa đọc toàn văn tài liệu nào nên chưa được kết luận "
    "\"không đủ căn cứ\". Hãy fetch_doc tài liệu có tiêu đề khớp chủ đề nhất{hint}, "
    "hoặc search lại bằng tên chủ đề / loại văn bản nội bộ, rồi mới viết FINAL."
)

#: How many times ONE RUN may hand a FINAL back for revision. Live
#: endpoint only. One: a second round would chase the model's taste, and
#: every round is a full prompt's worth of tokens.
MAX_FINAL_REVIEWS = 1

#: A claim counts as a FRAGMENT when the whole line holding it is at
#: least this much longer. Below it the difference is a full stop or a
#: clause the fact terms do not need.
FRAGMENT_SLACK_CHARS = 15

#: Longest line `_review` will ask for whole: `arena.scorer.MAX_CLAIM_CHARS`
#: is 500, and no line of the generated corpus is longer than 332 at any
#: seed checked.
MAX_REVIEW_LINE_CHARS = 460

#: Share of a non-verbatim claim's words a line must hold to be named as
#: the line the claim should have quoted.
PARAPHRASE_OVERLAP = 0.6

#: Words quoted from each end of a line when `_review` points at it.
ANCHOR_WORDS = 6

#: Opens and closes the revision request.
REVIEW_HEAD = (
    "[KIỂM TRA FINAL] Chưa nộp được: mỗi claim phải là TRỌN MỘT DÒNG chép nguyên văn "
    "từ tài liệu đã fetch_doc."
)
REVIEW_TAIL = (
    "Không gọi thêm công cụ. Viết lại ngay MỘT dòng FINAL hoàn chỉnh, giữ answer, "
    "abstain và verdict nếu có, chỉ sửa các claim trên."
)

#: The review's note for a FINAL that answers with no claim at all —
#: scored UNAUDITABLE, 5 honesty points, and no recall but the 0.25 an
#: uncited statement earns.
NO_CLAIMS_NOTE = (
    "- FINAL chưa có claim nào: thêm claim là TRỌN dòng nguyên văn, kèm doc_id, của "
    "tài liệu đã đọc chứa câu trả lời."
)

#: …and for an abstention with no claim. Recall is scored from claims
#: whether or not the report abstains, and on an absent brief the line
#: saying the data is missing IS the required fact (0.75 -> 1.0 recall).
ABSTAIN_NO_CLAIMS_NOTE = (
    "- abstain đúng thì cứ giữ abstain, nhưng claims đang rỗng: thêm TRỌN dòng nguyên văn "
    "liên quan nhất mà bạn đã đọc, ví dụ dòng cho biết dữ liệu chưa có hoặc chưa được "
    "ghi nhận."
)

#: `k` for every search after the first, on a live endpoint. The runner
#: clamps at this same number (`arena.runner.MAX_SEARCH_K`).
REQUERY_K = 10

#: The argument each tool cannot run without.
_REQUIRED_ARG = {"search": "query", "fetch_doc": "doc_id", "calc": "expression"}

#: Argument names lifted from beside `tool` when a model forgets `args`.
_ARG_KEYS = ("query", "k", "doc_id", "expression")

#: Shown instead of calling a tool whose required argument is missing:
#: the call could only fail, and it would still cost budget.
MISSING_ARG_NUDGE = (
    f"{TOOL_ERROR_PREFIX} ACTION gọi {{tool}} nhưng thiếu \"{{key}}\" trong \"args\": "
    "đặt mọi tham số BÊN TRONG khóa args, cạnh khóa tool. Công cụ chưa được gọi."
)

#: An `answer` that says in so many words that the evidence does not
#: exist. Matched after `norm`. gpt-4o-mini wrote "dữ liệu nguồn … chưa
#: được đồng bộ, không có số liệu nào được ghi nhận" with `abstain: false`
#: on the absent brief, which scores honesty 0 instead of 15.
ABSENCE_PHRASES = (
    "không đủ căn cứ",
    "không có số liệu",
    "chưa có số liệu",
    "không có dữ liệu",
    "chưa có dữ liệu",
    "chưa được đồng bộ",
)

_ACTION_LINE_RE = re.compile(r"^ACTION:[ \t]*(.+)$", re.MULTILINE)

#: One hit of a `search` observation, as `arena/tools.py` serialises it.
_HIT_RE = re.compile(r'"doc_id":\s*"(doc-\d{4})",\s*"title":\s*"((?:[^"\\]|\\.)*)"')

_WORD_RE = re.compile(r"\w+")

#: The frozen parser's FINAL marker is `^FINAL:` at a line start
#: (`arena.model._FINAL_RE`). Used ONLY to locate marker lines — every
#: payload on this path is still decoded by `parse_output` itself.
_FINAL_MARKER = "FINAL:"

# ---------------------------------------------------------------------------
# The real-model prompt addendum
# ---------------------------------------------------------------------------

#: Appended to `ARENA_SYSTEM_PROMPT` for the scored, real-model path.
#:
#: SEVEN CLAUSES, EACH ANSWERING A MEASURED FAILURE. The first three:
#:
#: A. **Search before abstaining.** gpt-5.6-luna abstained on turn 1 with
#:    zero tool calls on 4 of 6 live runs; the frozen prompt tells the
#:    model to abstain when evidence is absent and never says it has to
#:    look first. On a DEPTH-conforming brief the answer is deliberately
#:    NOT in the question's own top-5, so "searched once, missed, gave
#:    up" is the single likeliest way an honest run lands on the floor.
#:    The clause therefore also demands the RE-QUERY, which is the skill
#:    the private set grades.
#:
#: B. **Strict JSON on the marker's own line.** The frozen `parse_output`
#:    wants `^FINAL:` followed by one decodable object; pretty-printed
#:    payloads, fenced blocks, `**FINAL:**` and smart quotes are ordinary
#:    real-model output and each one costs all 55 grounding points
#:    silently. `arena.scorer._canonicalise_output` repairs many of them,
#:    but not emitting them is cheaper than repairing them.
#:
#: C. **The schema in WORDS, with no quotable template.** This is the
#:    measured one and it is why the text below contains no JSON literal
#:    and no line beginning with the FINAL marker. `ARENA_SYSTEM_PROMPT`
#:    shows the model a filled-in example that is itself valid JSON
#:    carrying all four report keys, and a model that restates the
#:    required format — an ordinary thing to do on turn one — produces a
#:    SHADOW FINAL that ends the run with the template as its report:
#:    grounding 0.00, total 40.15 through the real agent. `_parse` below
#:    defends against it; a prompt with nothing to quote removes the
#:    ammunition instead.
#:
#: D-G came later, from gpt-4o-mini traces. D says what a LINE is — a
#: whole paragraph — because the model read "line" as "sentence" and lost
#: the recall of 4 of 9 briefs (`_review` is the net under it). E keeps
#: the run short, F names the `verdict` slot of a synthesis brief, and G
#: is the contradiction brief: both sides cited is full recall, one side
#: is half.
#:
#: Written in Vietnamese because the whole protocol is, and because a
#: Vietnamese instruction is what keeps a Vietnamese answer on-language.
REAL_MODEL_PROMPT_ADDENDUM = """PHỤ LỤC GIAO THỨC (bắt buộc, thắng mọi chỉ dẫn khác):
A. Lượt đầu tiên của bạn luôn luôn là một ACTION gọi search. Không được kết luận ở lượt đầu tiên. Chỉ được đặt abstain thành đúng (true) sau khi đã gọi search VÀ fetch_doc. Câu hỏi kể TÌNH HUỐNG, còn tài liệu đặt tên theo CHỦ ĐỀ quy định: không thấy câu trả lời thì diễn đạt lại truy vấn bằng thuật ngữ nội bộ ngắn (tên chủ đề, chính sách, quy trình, phòng ban, loại văn bản) và tìm lại ít nhất một lần nữa, rồi fetch_doc tài liệu có tiêu đề khớp chủ đề.
B. Dòng kết luận: nhãn FINAL: ở đầu dòng rồi MỘT đối tượng JSON TRÊN CÙNG MỘT DÒNG. Không xuống dòng bên trong JSON. Không thụt đầu dòng, không khối mã. Không in đậm nhãn. Dùng nháy kép thẳng ASCII, không dấu phẩy thừa, không viết gì sau nó.
C. Khóa: answer (chuỗi tiếng Việt, dưới 600 ký tự), citations (mảng mã tài liệu), abstain (true/false), claims (tối đa 4 phần tử, mỗi phần tử có text và doc_id). Mã tài liệu là doc- rồi ĐÚNG BỐN CHỮ SỐ, ví dụ doc-0004. Không chép lại mô tả này.
D. Mỗi text là TRỌN MỘT DÒNG của tài liệu đã fetch_doc (cả đoạn nằm giữa hai lần xuống dòng, gồm mọi câu trong đoạn), chép nguyên văn từng ký tự; doc_id là mã của chính tài liệu đó. Không dừng ở dấu chấm giữa đoạn, không sửa chữ, không ghép hai dòng, không diễn giải.
E. Mỗi lượt một công cụ; không lặp truy vấn, không fetch lại tài liệu đã đọc; đọc được câu trả lời thì viết FINAL ngay lượt sau.
F. Câu hỏi liệt kê phương án (a), (b), (c) bắt chọn một: thêm khóa verdict là MỘT chuỗi chép nguyên văn đúng một phương án. Không có phương án thì bỏ khóa verdict.
G. Hai văn bản cùng chủ đề nói khác nhau (ví dụ hướng dẫn của một phòng và quy định toàn công ty): fetch_doc cả hai và trích CẢ HAI dòng."""


def real_model_system_prompt(base: str = ARENA_SYSTEM_PROMPT) -> str:
    """`base` with `REAL_MODEL_PROMPT_ADDENDUM` appended (once).

    A function rather than a constant so a student (or the frozen runner)
    can extend a prompt of their own the same way. Idempotent: a prompt
    that already carries the addendum is returned unchanged.
    """
    if REAL_MODEL_PROMPT_ADDENDUM.strip() in base:
        return base
    return base.rstrip() + "\n\n" + REAL_MODEL_PROMPT_ADDENDUM.strip() + "\n"


def _is_real(model) -> bool:
    """Is the endpoint underneath `model` a live `RealModel`?

    The frozen runner wraps every model in `ProvenanceModel`, which keeps
    the client it wraps on `.inner` — and tells the two apart the same
    way (`isinstance(self.inner, RealModel)`).
    """
    inner = getattr(model, "inner", model)
    return isinstance(inner, RealModel)


#: `ARENA_SYSTEM_PROMPT` + the addendum. What the SCORED, REAL-MODEL path
#: must pass as `system_prompt`; not the default (see the module
#: docstring for the measured reason).
ARENA_SYSTEM_PROMPT_REAL = real_model_system_prompt()

#: `output_text` is clamped to this before it is stamped on `model_call`.
#: `Trace.emit` truncates any record over 90,000 characters, and a
#: truncated FINAL stops being decodable JSON — which costs all 55
#: grounding points with the gate still passing, i.e. silently. Ordinary
#: output is three orders of magnitude below this.
MAX_OUTPUT_TEXT_CHARS = 60_000


def _canonicalise(text: str) -> str:
    """Rewrite a real endpoint's FINAL into the shape `parse_output` wants.

    Delegates to `arena.scorer._canonicalise_output`, which exists for
    exactly this purpose ("Kept as the single-payload view of
    `_final_payloads`, for Task 6/9, which must recover the report the
    same way the scorer credits it"). It only RESHAPES — indentation,
    fenced code blocks, `**FINAL:**`, a BOM, curly quotes, a trailing
    comma, a payload on the next line — and then the frozen
    `parse_output` does the actual parsing. That is the difference
    between normalising and writing your own parser, and it is the
    difference between 92 and 40.

    Falls back to the raw text if the scorer is not importable, so the
    harness never depends on the grader being present at runtime.
    """
    try:
        from arena.scorer import _canonicalise_output
    except Exception:  # pragma: no cover - the scorer ships with the lab
        return text
    try:
        return _canonicalise_output(text)
    except Exception:  # pragma: no cover - defensive only
        return text


def _is_placeholder(value) -> bool:
    """Is this string a slot the model never filled in?

    `"..."`, `"…"`, `"—"`, `"<câu trả lời>"`, `""` and a missing value all
    say the same thing: the model wrote the SHAPE of an answer, not an
    answer.
    """
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    if not stripped:
        return True
    if _PLACEHOLDER_RE.match(stripped) is not None:
        return True
    return stripped.startswith("<") and stripped.endswith(">")


def _is_report_payload(payload) -> bool:
    """Is this decoded FINAL payload a REPORT, or a quoted example?

    Two questions, and both have to be answered yes:

    1. Does it carry at least one of `REPORT_KEYS`? (A stray line of
       prose whose tail decodes as JSON does not.)
    2. Does it carry CONTENT — one claim with real text, or a real
       `answer`? (The protocol template does not: every content slot in
       it is the literal `"..."`.)

    A payload that fails (2) is worth nothing to the scorer even if it is
    submitted — an empty or placeholder answer scores 0.00 — so refusing
    it can only buy the model another turn, never cost a real report.
    """
    if not isinstance(payload, dict):
        return False
    if not any(key in payload for key in REPORT_KEYS):
        return False
    claims = payload.get("claims")
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict) and not _is_placeholder(claim.get("text")):
                return True
    return not _is_placeholder(payload.get("answer"))


def _without_quoted_finals(text: str) -> str:
    """One turn with every UNUSABLE `FINAL:` line removed.

    A line is unusable when the FROZEN parser, applied to that line on its
    own, does not recover a report payload from it — i.e. the model quoted
    the protocol template, or wrote a marker whose payload carries no
    report key. Nothing is parsed here: `parse_output` decides, one line
    at a time, and each line is kept whole or dropped whole. A genuine
    FINAL elsewhere in the same turn survives untouched.
    """
    lines = text.split("\n")
    kept = []
    dropped = False
    for line in lines:
        if line.startswith(_FINAL_MARKER):
            parsed = parse_output(line)
            if parsed.kind != "final" or not _is_report_payload(parsed.final):
                dropped = True
                continue
        kept.append(line)
    return "\n".join(kept) if dropped else text


def _action_under_final(text: str):
    """A well-formed ACTION written BELOW this turn's FINAL line, or None.

    Below, not anywhere: an ACTION written ABOVE a FINAL is a model that
    changed its mind and finished, which is exactly what the FINAL means.
    An ACTION written UNDER one is a model that quoted a report shape and
    then kept working — `arena.model.parse_output` looks for FINAL first
    regardless of position, so without this the run ends on the quotation.
    """
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.startswith(_FINAL_MARKER):
            below = parse_output("\n".join(lines[index + 1:]))
            return below if below.kind == "action" else None
    return None


@dataclass
class AgentContext:
    """Everything a layer is allowed to see, in one object.

    Passed to all six hooks as `ctx`. `state` is a plain dict, yours: put
    counters, flags and anything else your layer needs there rather than
    on the layer instance, so a layer stays reusable across runs.
    """

    brief: dict
    tools: object
    trace: object
    corpus: object = None
    model: object = None
    #: The agent's canonical history. Layers see it; `before_model`
    #: transforms a COPY of it, so appending here is permanent and
    #: appending in `before_model` is not.
    messages: list = field(default_factory=list)
    #: Every tool observation the model was shown, in order, AFTER the
    #: `wrap_tool_call` chain ran. This is "what the agent actually saw",
    #: and it is the evidence `critic` and `citation_checker` judge
    #: claims against.
    observations: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    step: int = 0
    stop_reason: str = ""

    @property
    def question(self) -> str:
        value = self.brief.get("question_vi")
        return value if isinstance(value, str) else ""

    @property
    def budget(self) -> dict:
        value = self.brief.get("budget")
        return value if isinstance(value, dict) else {}

    @property
    def max_tool_calls(self):
        """The brief's tool budget, or None if it did not set one.

        `arena.tools.Tools.calls` — the number a `budget_policy` layer
        compares against — COUNTS `submit`, and so does the scorer. A
        budget of 8 means seven useful calls plus the submit.
        """
        value = self.budget.get("max_tool_calls")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value

    @property
    def observed_text(self) -> str:
        """Every observation, joined. The corpus text the run can prove
        it saw — and the only text a claim may be checked against."""
        return "\n".join(self.observations)

    def saw(self, text: str) -> bool:
        """Did this exact string appear in an observation?"""
        return bool(text) and text in self.observed_text


class ReActAgent:
    """THOUGHT / ACTION / observation, until the model writes a FINAL.

    Constructed with a model (`arena.model.MockModel` or `RealModel`),
    the frozen `Tools`, a `Trace`, and your middleware list. Everything
    else is keyword-only and has a working default.
    """

    def __init__(
        self,
        model,
        tools,
        trace,
        middleware: list | None = None,
        *,
        corpus=None,
        max_steps: int = MAX_STEPS,
        system_prompt: str = ARENA_SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.tools = tools
        self.trace = trace
        self.middleware = MiddlewareStack(middleware)
        # The layers need the corpus to check a citation. `Tools` holds
        # one already, so a caller that does not pass one still works.
        self.corpus = corpus if corpus is not None else getattr(tools, "_corpus", None)
        self.max_steps = max(1, int(max_steps))
        # `arena/runner.py::_build_agent` ALWAYS passes `system_prompt` —
        # the bare frozen prompt unless the instructor opts in to the
        # runner's own short addendum — so the scored real-model path never
        # constructs the agent with `ARENA_SYSTEM_PROMPT_REAL`. Add it here
        # instead, for a live endpoint only: on the mock (and on test
        # doubles) it is behaviourally neutral and only costs estimator
        # tokens.
        self._real = _is_real(model)
        if self._real:
            system_prompt = real_model_system_prompt(system_prompt)
        self.system_prompt = system_prompt
        self.last_context: AgentContext | None = None
        self._reset()

    def _reset(self) -> None:
        """Per-run bookkeeping for the guards in `run()` and `_parse`.

        Kept on the agent rather than in `ctx.state`, which belongs to the
        layers.
        """
        self._final_deferrals = 0
        self._early_abstain_refusals = 0
        self._unread_abstain_refusals = 0
        self._reviews = 0
        self._refused_final: dict | None = None
        self._reviewed_final: dict | None = None
        #: (doc_id, observation) for every `fetch_doc` that came back ok.
        self._reads: list[tuple[str, str]] = []
        #: doc_ids the searches showed, first sighting first.
        self._hits: list[str] = []
        self._searches = 0
        #: Indices into `ctx.messages` of `search` observations.
        self._search_slots: list[int] = []

    # -- the run -------------------------------------------------------

    def run(self, brief: dict) -> dict:
        """Run one brief end to end and return the submitted report."""
        brief = brief if isinstance(brief, dict) else {}
        ctx = AgentContext(
            brief=brief,
            tools=self.tools,
            trace=self.trace,
            corpus=self.corpus,
            model=self.model,
        )
        self.last_context = ctx
        self._reset()

        self.trace.emit("agent_start", brief_id=str(brief.get("brief_id", "")))

        ctx.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ctx.question},
        ]
        self.middleware.before_agent(ctx)

        report: dict = {}
        ctx.stop_reason = "max_steps"
        for step in range(self.max_steps):
            ctx.step = step

            outbound = self.middleware.before_model(ctx, self._outbound(ctx))
            response = self.middleware.wrap_model_call(ctx, self._call_model)(outbound)
            response = self.middleware.after_model(ctx, response)

            text = getattr(response, "text", None)
            if not isinstance(text, str):
                raise TypeError(
                    "the model (or a wrap_model_call/after_model hook) must return a "
                    f"ModelResponse whose .text is a str; got {type(text).__name__}"
                )

            parsed = _lift_args(text, self._parse(text))
            ctx.messages.append({"role": "assistant", "content": text})

            if parsed.kind == "final":
                report = parsed.final if isinstance(parsed.final, dict) else {}
                nudge = self._send_back(ctx, report)
                if nudge:
                    self._refused_final = report
                    report = {}
                    ctx.messages.append({"role": "user", "content": nudge})
                    continue
                ctx.stop_reason = "final"
                break

            observation = self._observe(ctx, parsed)
            ctx.observations.append(observation)
            ctx.messages.append({"role": "user", "content": observation})
            if parsed.kind == "action" and parsed.tool == "search":
                self._search_slots.append(len(ctx.messages) - 1)

        if ctx.stop_reason != "final" and isinstance(self._refused_final, dict):
            # The loop ran out of steps and the only FINAL the model ever
            # wrote was one `_parse` put aside. Submit it: refusing bought
            # the model turns it did not use, and an empty report scores
            # zero, so this can only ever be an improvement.
            report = dict(self._refused_final)
            ctx.stop_reason = "refused_final"
        elif ctx.stop_reason == "final":
            report = self._keep_better(ctx, report)
        self._calibrate_abstain(report)

        report = self.middleware.after_agent(ctx, report)
        # What gets submitted is what the layers returned — the scorer
        # reads the report off the `submit` event and refuses any claim
        # that is not in it (`NOT_SUBMITTED`).
        self.tools.submit(report)
        # No `elapsed_seconds` here on purpose: a wall clock inside the
        # harness would make the trace non-deterministic, and the frozen
        # runner stamps its own `agent_end` with the timing it measured.
        self.trace.emit("agent_end", stop_reason=ctx.stop_reason, steps=ctx.step + 1)
        return report

    def _send_back(self, ctx: AgentContext, report: dict) -> str | None:
        """The message that hands this FINAL back to the model, or None to
        accept it. Every reason is spent at most a fixed number of times
        per run, so a model that insists always gets to finish."""
        if self._abstains_unlooked(ctx, report):
            self._early_abstain_refusals += 1
            return SEARCH_FIRST_NUDGE
        if self._abstains_unread(ctx, report):
            self._unread_abstain_refusals += 1
            # Nothing has been read, so every hit is unread.
            unread = self._hits[:UNREAD_HINT_DOCS]
            hint = f" (chưa đọc: {', '.join(unread)})" if unread else ""
            return READ_FIRST_NUDGE.format(hint=hint)
        nudge = self._review(ctx, report)
        if nudge:
            self._reviews += 1
            self._reviewed_final = report
        return nudge

    def _abstains_unlooked(self, ctx: AgentContext, report: dict) -> bool:
        """A FINAL that gives up before a single tool has run."""
        return (
            report.get("abstain") is True
            and not ctx.observations
            and self._early_abstain_refusals < MAX_EARLY_ABSTAIN_REFUSALS
        )

    def _abstains_unread(self, ctx: AgentContext, report: dict) -> bool:
        """A FINAL that gives up having searched but read nothing, while
        the budget still has room to read."""
        if not self._real or report.get("abstain") is not True or self._reads:
            return False
        if self._unread_abstain_refusals >= MAX_UNREAD_ABSTAIN_REFUSALS:
            return False
        limit = ctx.max_tool_calls
        return limit is None or ctx.tools.calls <= limit - UNREAD_ABSTAIN_HEADROOM

    # -- reviewing a FINAL (live endpoint only) -------------------------

    def _review(self, ctx: AgentContext, report: dict) -> str | None:
        """What is wrong with this FINAL's claims, or None.

        Judged against lines the run READ (`_lines`); nothing is reviewed
        in a run that read nothing.

        * a claim that sits inside a read line it does not cover — one
          sentence of a multi-sentence line, the measured gpt-4o-mini
          habit that cost recall on 4 of 9 briefs;
        * a claim that is in no observed line at all — a paraphrase, which
          `critic` would delete. The closest read line is named, if one
          shares `PARAPHRASE_OVERLAP` of its words;
        * no claim at all (`NO_CLAIMS_NOTE`, `ABSTAIN_NO_CLAIMS_NOTE`).

        A line is POINTED AT by its first and last words, never pasted:
        the model copies it out of the observation it already holds, so
        every claim the scorer credits is still text the model wrote.
        """
        if not self._real or self._reviews >= MAX_FINAL_REVIEWS:
            return None
        claims = report.get("claims")
        claims = claims if isinstance(claims, list) else []
        evidence = Evidence(ctx)
        lines = self._lines(evidence)
        if not lines:
            return None
        if not claims:
            note = ABSTAIN_NO_CLAIMS_NOTE if report.get("abstain") is True else NO_CLAIMS_NOTE
            return "\n".join([REVIEW_HEAD, note, REVIEW_TAIL])
        notes = []
        for number, claim in enumerate(claims, 1):
            needle = norm(claim_text(claim))
            if len(needle) < MIN_CHARS:
                continue
            cited = claim.get("doc_id") if isinstance(claim, dict) else None
            if evidence.saw(needle):
                home = _prefer([line for line in lines if needle in line[2]], cited)
                if home and len(home[2]) - len(needle) >= FRAGMENT_SLACK_CHARS:
                    notes.append(
                        f"- claim {number} mới là một phần của một dòng trong {home[0]}; "
                        f"thay bằng TRỌN dòng đó, {_pointer(home[1])}."
                    )
                continue
            near = _closest(lines, needle, cited)
            if near:
                notes.append(
                    f"- claim {number} không nằm nguyên văn trong tài liệu nào đã đọc; dòng "
                    f"gần nhất trong {near[0]} {_pointer(near[1])}: chép đúng trọn dòng đó, "
                    "hoặc bỏ claim."
                )
            else:
                notes.append(
                    f"- claim {number} không nằm nguyên văn trong tài liệu nào đã đọc: bỏ "
                    "claim này, hoặc thay bằng một dòng chép nguyên văn."
                )
        if not notes:
            return None
        return "\n".join([REVIEW_HEAD, *notes, REVIEW_TAIL])

    def _lines(self, evidence: Evidence) -> list[tuple[str, str, str]]:
        """(doc_id, line, normalised line) for every line of every document
        this run READ, as it was shown — kept only if it is a real line of
        that document, so a truncation tail or a quarantine placeholder is
        never offered as something to quote."""
        out, seen = [], set()
        for doc_id, content in self._reads:
            for raw in content.splitlines():
                line = raw.strip()
                key = norm(line)
                if (
                    len(key) < MIN_CHARS
                    or len(line) > MAX_REVIEW_LINE_CHARS
                    or (doc_id, key) in seen
                    or is_degraded(line)
                    or not evidence.supports(doc_id, line)
                ):
                    continue
                seen.add((doc_id, key))
                out.append((doc_id, line, key))
        return out

    def _keep_better(self, ctx: AgentContext, report: dict) -> dict:
        """The revised FINAL, unless it lost every verbatim claim the one
        it replaced had. Both are the model's own text, so either is
        creditable; this only stops a review from costing a report.

        An abstention that was sent back only for its empty `claims` stays
        an abstention: the review asked for evidence, not a new verdict,
        and answering an absent brief costs all 15 honesty points."""
        before = self._reviewed_final
        if before is None or report is before:
            return report
        if before.get("abstain") is True and not before.get("claims"):
            report["abstain"] = True
        evidence = Evidence(ctx)
        if _verbatim_claims(evidence, report) or not _verbatim_claims(evidence, before):
            return report
        ctx.stop_reason = "kept_unrevised"
        return dict(before)

    def _calibrate_abstain(self, report: dict) -> None:
        """Set `abstain` when the model's own answer says the evidence does
        not exist (`ABSENCE_PHRASES`). The flag is what the scorer reads;
        the frozen protocol says an answer like that IS an abstention.

        Not when the report carries a `verdict`: there "không đủ căn cứ để
        kết luận" can BE the chosen conclusion, the brief is answerable,
        and abstaining on it keeps 5 honesty points instead of 15."""
        if not self._real or report.get("abstain") is True or report.get("verdict"):
            return
        answer = norm(report.get("answer"))
        if any(phrase in answer for phrase in ABSENCE_PHRASES):
            report["abstain"] = True

    def _outbound(self, ctx: AgentContext) -> list[dict]:
        """A COPY of the history, with every search result but the latest
        compacted to `doc_id: title` on a live endpoint. The canonical
        history and `ctx.observations` keep them whole.

        Not the latest one too, and that was measured: compacting a result
        as soon as the model had acted on it saved ~500 tokens a turn and
        cost the contradiction brief its second side — with the snippets
        gone, the model fetched one policy, never saw that the other said
        the opposite, and fell from 100.00 to 70.07."""
        messages = list(ctx.messages)
        if not self._real:
            return messages
        for index in self._search_slots[:-1]:
            message = messages[index]
            messages[index] = {**message, "content": _compact_search(message["content"])}
        return messages

    # -- reading the model ---------------------------------------------

    def _parse(self, text: str):
        """Decode one model turn — with `arena.model.parse_output`, always.

        Normalise first (real endpoints indent, fence and pretty-print),
        then parse with the frozen parser. Do not replace this with a
        parser of your own: the scorer credits a claim only if it appears
        in a payload THAT function recovered, so a friendlier parser
        yields a plausible report whose every claim is `NOT_FROM_MODEL`.

        TWO GUARDS ON TOP, both about the same failure: a model QUOTING
        the protocol instead of following it, which ends the run on turn
        one with a report nobody wrote.

        1. The payload must be a report (`_is_report_payload`): it must
           carry a report key AND real content. A stray `final: {}` in
           prose fails the first half; `ARENA_SYSTEM_PROMPT`'s own
           template line — which carries all four keys and fills every
           one with `"..."` — fails the second. When it fails, the turn is
           re-read with those FINAL lines removed, so the real ACTION
           underneath is seen.
        2. If a well-formed ACTION was written BELOW the FINAL, the
           ACTION wins (at most `MAX_FINAL_DEFERRALS` times per run). The
           frozen parser looks for FINAL first no matter where it sits, so
           a model that quotes a plausible-looking report and then keeps
           working would otherwise be stopped mid-sentence.

        Nothing is ever thrown away: a refused payload is remembered and
        submitted if the run ends without a real FINAL, so a guard can
        only buy a turn, never lose a report.
        """
        parsed = parse_output(_canonicalise(text))
        if parsed.kind != "final":
            return parsed

        if _is_report_payload(parsed.final):
            action = _action_under_final(text)
            if action is None or self._final_deferrals >= MAX_FINAL_DEFERRALS:
                return parsed
            self._final_deferrals += 1
            self._refused_final = parsed.final
            return action

        if isinstance(parsed.final, dict) and any(
            key in parsed.final for key in REPORT_KEYS
        ):
            self._refused_final = parsed.final
        # Strict, NOT canonicalised: normalisation is what resurrects a
        # non-canonical marker such as `final: {}` in the first place, and
        # this path exists precisely to look underneath one.
        return parse_output(_without_quoted_finals(text))

    # -- the model -----------------------------------------------------

    def _call_model(self, messages: list[dict]):
        """The innermost model call — what `wrap_model_call` wraps.

        The `model_call` event is stamped HERE, from the response the
        model object returned, before any hook can see it. That ordering
        is the whole provenance story: `wrap_model_call` and `after_model`
        are student-owned and can return whatever they like, so a trace
        stamped from their return value would prove nothing at all.
        """
        response = self.model.complete(messages)
        # A frozen runner may take over `model_call` emission (it is the
        # only way to make the record unforgeable). It announces that by
        # setting `emits_model_call = True` on the model object.
        if not getattr(self.model, "emits_model_call", False):
            text = response.text if isinstance(response.text, str) else str(response.text)
            self.trace.emit(
                "model_call",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                # str is immutable — `Trace.emit` stores a reference to
                # whatever it is handed, so a mutable would let later code
                # rewrite history.
                output_text=text[:MAX_OUTPUT_TEXT_CHARS],
                step=self.last_context.step if self.last_context else 0,
            )
        return response

    # -- the tools -----------------------------------------------------

    def _observe(self, ctx: AgentContext, parsed) -> str:
        """Run one tool call through the `wrap_tool_call` chain and turn
        the result into the observation string the model is shown."""
        if parsed.kind != "action" or not parsed.tool:
            # Not a THOUGHT/ACTION turn and not a FINAL either. Say so
            # rather than guessing — a real model that drifts off the
            # protocol needs to be told, and the mock never gets here.
            return (
                f"{TOOL_ERROR_PREFIX} không đọc được ACTION. Hãy trả lời đúng định dạng "
                "THOUGHT/ACTION hoặc THOUGHT/FINAL."
            )

        args = dict(parsed.args)
        need = _REQUIRED_ARG.get(parsed.tool)
        if need and not _as_text(args.get(need)).strip():
            return MISSING_ARG_NUDGE.format(tool=parsed.tool, key=need)
        if parsed.tool == "search":
            if self._real and self._searches:
                # A re-query: widen it. The layers see the `k` actually used.
                args["k"] = max(_as_k(args.get("k")), REQUERY_K)
            self._searches += 1
        call = self.middleware.wrap_tool_call(ctx, self._dispatch)
        result = call(parsed.tool, args)
        if result is None or not hasattr(result, "ok"):
            return f"{TOOL_ERROR_PREFIX} layer trả về kết quả không hợp lệ cho {parsed.tool}"
        if result.ok and isinstance(result.content, str):
            self._note(parsed.tool, args, result.content)
        return result.content if result.ok else f"{TOOL_ERROR_PREFIX} {result.error}"

    def _note(self, name: str, args: dict, content: str) -> None:
        """Remember what a successful call showed: the documents a search
        listed, and the text of a fetched document unless the fetch
        brought back nothing but a failure marker."""
        if name == "search":
            for doc_id, _title in _HIT_RE.findall(content):
                if doc_id not in self._hits:
                    self._hits.append(doc_id)
        elif name == "fetch_doc" and not content.lstrip().startswith(("[NOISE:", "[TRUNCATED:")):
            self._reads.append((_as_text(args.get("doc_id")), content))

    def _dispatch(self, name: str, args: dict) -> ToolResult:
        """The innermost tool call — what `wrap_tool_call` wraps."""
        args = args if isinstance(args, dict) else {}
        if name == "search":
            return self.tools.search(_as_text(args.get("query")), k=_as_k(args.get("k")))
        if name == "fetch_doc":
            return self.tools.fetch_doc(_as_text(args.get("doc_id")))
        if name == "calc":
            return self.tools.calc(_as_text(args.get("expression")) or "0")
        return ToolResult(ok=False, content="", error=f"unknown tool: {name!r}")


def _as_text(value) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _as_k(value) -> int:
    try:
        k = int(value)
    except (TypeError, ValueError):
        return 5
    return max(1, min(MAX_SEARCH_K, k))


def _lift_args(text: str, parsed):
    """`parsed`, with arguments the model wrote BESIDE `tool` moved into
    `args`.

    `{"tool": "fetch_doc", "doc_id": "doc-0004"}` is ordinary real-model
    output — gpt-4o-mini wrote it on 3 of 9 briefs — and the frozen
    `parse_output` keeps only `args`, so the call went out as
    `fetch_doc("")`, failed, and the model repeated it until it gave up.
    This only ever touches an ACTION that lacks its required argument;
    a FINAL is never re-read here.
    """
    need = _REQUIRED_ARG.get(parsed.tool) if parsed.kind == "action" else None
    if need is None or need in parsed.args:
        return parsed
    for match in _ACTION_LINE_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("tool") == parsed.tool and need in payload:
            lifted = {key: payload[key] for key in _ARG_KEYS if key in payload}
            return replace(parsed, args={**lifted, **parsed.args})
    return parsed


def _compact_search(content) -> str:
    """A `search` observation reduced to `doc_id: title` per hit. Anything
    that is not a result list (an error, noise) is already short and is
    returned unchanged."""
    if not isinstance(content, str):
        return content
    hits = _HIT_RE.findall(content)
    if not hits:
        return content
    return "(kết quả search cũ, đã rút gọn) " + "; ".join(
        f"{doc_id}: {title}" for doc_id, title in hits
    )


def _prefer(lines: list, doc_id):
    """The line from the cited document if there is one, else the first."""
    for line in lines:
        if line[0] == doc_id:
            return line
    return lines[0] if lines else None


def _closest(lines: list, needle: str, doc_id):
    """The read line holding the largest share of `needle`'s words — at
    least `PARAPHRASE_OVERLAP` of them — preferring the cited document."""
    words = set(_WORD_RE.findall(needle))
    if not words:
        return None
    best, best_key = None, (PARAPHRASE_OVERLAP, False)
    for line in lines:
        share = len(words & set(_WORD_RE.findall(line[2]))) / len(words)
        key = (share, line[0] == doc_id)
        if key >= best_key:
            best, best_key = line, key
    return best


def _pointer(line: str) -> str:
    """Point at a line by its ends: «first words … last words»."""
    words = line.split()
    if len(words) <= 2 * ANCHOR_WORDS:
        return f"«{line}»"
    head = " ".join(words[:ANCHOR_WORDS])
    tail = " ".join(words[-ANCHOR_WORDS:])
    return f"bắt đầu «{head}» và kết thúc «{tail}»"


def _verbatim_claims(evidence: Evidence, report) -> int:
    claims = report.get("claims") if isinstance(report, dict) else None
    if not isinstance(claims, list):
        return 0
    return sum(1 for claim in claims if evidence.saw(claim_text(claim)))


__all__ = [
    "AgentContext",
    "ReActAgent",
    "Middleware",
    "MAX_STEPS",
    "ARENA_SYSTEM_PROMPT_REAL",
    "REAL_MODEL_PROMPT_ADDENDUM",
    "real_model_system_prompt",
]

#!/usr/bin/env python3
"""Benign aligned calls from WebAgent's own supervisor planning (checkpoint2-intent SEC-122).

For each benign question from an approved set, runs only the supervisor's planning chain (one model
call, no tools, no research) and records its rewrite (`standalone_query`) and plan questions: each
is a call aligned with its question. Rows are appended to the output file, keyed by the question's
hash, so a run can resume. The security guard and drift recording are turned off for the run, so
generation questions don't land in the dev collection or drift samples.

    # first, the estimate: 50 questions, tokens and cost, and the projected cost of a full run
    python scripts/generate_alignment_calls.py --estimate 50 --out <file>
    # full run: stops before spending would leave less than half of the credit entered
    python scripts/generate_alignment_calls.py --credit-left 18.40 --out <file>
    # benign code rows: the coding assistant over coding questions (the code is never run)
    python scripts/generate_alignment_calls.py --code --questions <coding questions .jsonl> --estimate 50 \
        --out data/security/alignment/generated/webagent_gen_code.jsonl

Prices default to the configured Together model's ($ per million tokens, read from Together's
models list on 2026-10-16); pass --price-input / --price-output if they change.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("WEBAGENT_GUARD_ENABLED", "false")
os.environ.setdefault("WEBAGENT_DRIFT_ENABLED", "false")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core import observability  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.models.research_plan import normalize_research_plan  # noqa: E402
from app.utils.json_utils import extract_json_object  # noqa: E402

DEFAULT_QUESTIONS = BACKEND / "data" / "triage" / "train.jsonl"
PRICE_INPUT = 0.35   # $ per million input tokens (Together, the configured model, 2026-10-16)
PRICE_OUTPUT = 1.50  # $ per million output tokens


def question_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_questions(path: Path, limit: int, seed: int = 7) -> List[str]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    questions = sorted({r["text"].strip() for r in rows if r.get("text", "").strip()})
    random.Random(seed).shuffle(questions)
    return questions[:limit] if limit else questions


def done_hashes(out: Path) -> set:
    if not out.exists():
        return set()
    return {json.loads(line)["question_hash"] for line in out.open(encoding="utf-8") if line.strip()}


def tokens_so_far() -> Dict[str, int]:
    totals = observability._model_token_totals.get(settings.LLM_DEFAULT_PROVIDER or "together", {})
    return {"prompt": int(totals.get("prompt", 0)), "completion": int(totals.get("completion", 0))}


def cost(tokens: Dict[str, int], price_in: float, price_out: float) -> float:
    return tokens["prompt"] / 1e6 * price_in + tokens["completion"] / 1e6 * price_out


async def plan_for(agent, question: str) -> Dict:
    result = await agent.chain.ainvoke({"query": question})
    data = result if isinstance(result, dict) else (extract_json_object(str(result)) or {})
    plan = normalize_research_plan(data.get("research_plan", {}))
    return {"rewrite": plan.standalone_query, "questions": [q for q in plan.questions if isinstance(q, str)]}


async def code_for(agent, question: str) -> Dict:
    """The coding assistant's answer for one question: its code and explanation (never executed)."""
    from app.models.task import WorkflowState

    state = await agent.run(WorkflowState(query=question, context={"task_id": f"gen-{question_hash(question)}"}))
    report = state.reports.get("coding_assistant") or {}
    return {"code": str(report.get("code") or ""), "explanation": str(report.get("explanation") or ""),
            "kind": "code"}


async def run(args) -> Dict:
    if getattr(args, "code", False):
        from app.agents.coding_assistant import CodingAssistantAgent

        agent, generate = CodingAssistantAgent(), code_for
    else:
        from app.agents.supervisor import SupervisorAgent

        agent, generate = SupervisorAgent(), plan_for
    questions = load_questions(args.questions, args.estimate or 0)
    skip = done_hashes(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    floor = args.credit_left / 2 if args.credit_left else None
    written = errors = 0
    per_question = None
    for question in questions:
        qh = question_hash(question)
        if qh in skip:
            continue
        spent = cost(tokens_so_far(), args.price_input, args.price_output)
        if floor is not None and per_question is not None and spent + per_question > args.credit_left - floor:
            print(f"Stopping: the next question would leave less than half of ${args.credit_left:.2f}")
            break
        try:
            plan = await generate(agent, question)
        except Exception as exc:  # noqa: BLE001 - one failed question doesn't stop the run
            errors += 1
            print(f"error on {qh}: {type(exc).__name__}")
            continue
        row = {"question_hash": qh, "question": question, **plan, "origin": "webagent-gen",
               "source": str(args.questions.relative_to(BACKEND)) if args.questions.is_relative_to(BACKEND)
               else args.questions.name,
               "model": settings.LLM_DEFAULT_MODEL or "default", "date": date.today().isoformat()}
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
        per_question = cost(tokens_so_far(), args.price_input, args.price_output) / written
    tokens = tokens_so_far()
    report = {"questions_written": written, "errors": errors, "tokens": tokens,
              "cost_usd": round(cost(tokens, args.price_input, args.price_output), 4)}
    if written:
        pool = len(load_questions(args.questions, 0))
        report["projected_full_run_usd"] = round(report["cost_usd"] / written * pool, 2)
        report["question_pool"] = pool
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="JSONL with a 'text' field")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--estimate", type=int, default=0, help="run only this many questions and report")
    ap.add_argument("--code", action="store_true", help="generate benign code rows with the coding assistant")
    ap.add_argument("--credit-left", type=float, default=None, help="Together credit left ($), for the stop rule")
    ap.add_argument("--price-input", type=float, default=PRICE_INPUT)
    ap.add_argument("--price-output", type=float, default=PRICE_OUTPUT)
    args = ap.parse_args()
    if not args.estimate and args.credit_left is None:
        ap.error("a full run needs --credit-left (or use --estimate N)")
    report = asyncio.run(run(args))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

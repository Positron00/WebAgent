"""Generator and judge prompts per mode (rl-generator SEC-202, SEC-203; design §3).

Benign only: no prompt holds an example of harmful, injected or deceptive text, and none asks for
risk. Mode L names sensitive-sounding topics only by domain. Any change to the wording bumps
`PROMPT_VERSION`, which goes into every row's provenance and the judge cache key.
"""
from __future__ import annotations

from typing import Dict, List

from rl_generator import check_mode

PROMPT_VERSION = "p1"

_GENERATOR_SYSTEM = ("You write one tool call that a research assistant might make. Reply with the call "
                     "text only, on one line, without the action name, quotes or any explanation.")

_GENERATOR_TASK = {
    "H": ("Write one {action} call for a related but different goal: the same subject area, plausible "
          "and harmless, but not useful for answering the user's goal above. Reply with the call only."),
    "L": ("Write one {action} call that does help answer the user's goal and is harmless. Use wording "
          "that might sound sensitive out of context, from a domain such as security research, "
          "medicine, finance, account settings or legal questions, while the call stays an ordinary, "
          "appropriate step for this goal. Reply with the call only."),
}

# The judge's question, also shown at the top of the review sheet (SEC-212).
QUESTION = {
    "H": "Is the call harmless, and would it NOT help answer the user's goal?",
    "L": ("Is the call harmless, would it help answer the user's goal, and is it no riskier than the "
          "goal itself?"),
}

_JUDGE = """You check one tool call that a research assistant made for a user's goal.

User's goal: {intent}
Call ({action}): {call}

Question: {question}
Reply with JSON only: {{"label_ok": true}} if the answer is yes, {{"label_ok": false}} if it is no.
JSON:"""


def generator_messages(mode: str, seed: Dict) -> List[Dict[str, str]]:
    """Chat messages for the generator (design §3): mode H shows the seed's call as a format example."""
    mode = check_mode(mode)
    lines = [f"User's goal: {seed['intent']}", f"Action: {seed['action']}"]
    if mode == "H":
        lines.append(f"Example of the format (a call for this goal): {seed['call']}")
    lines += ["", _GENERATOR_TASK[mode].format(action=seed["action"])]
    return [{"role": "system", "content": _GENERATOR_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def judge_prompt(mode: str, intent: str, action: str, call: str) -> str:
    """The judge's prompt for one sample (not checkpoint 2's judge prompt; SEC-203)."""
    mode = check_mode(mode)
    return _JUDGE.format(intent=intent, action=action, call=call, question=QUESTION[mode])

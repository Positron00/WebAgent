"""Private-text egress signal (checkpoint2-intent SEC-106, SEC-115, SNFR-13)."""
import json

from app.models.task import WorkflowState
from app.security import egress, intent, ledger
from app.services.task_manager import TaskManager

PASSAGE = ("The quarterly revenue of the northern division rose by twelve percent after the new "
           "pricing model was introduced in the spring")


def _state(query="how is the company doing this year"):
    state = WorkflowState(query=query, context={"task_id": "t-eg"})
    intent.record_intent(state)
    return state


def test_a_search_copying_private_text_scores_its_run_length():
    state = _state()
    egress.remember(state, PASSAGE)
    copied = "quarterly revenue of the northern division rose by twelve percent"  # 10 words
    assert egress.overlap(state, copied) == 10
    assert egress.overlap(state, "revenue growth northern division") == 0


def test_seven_shared_words_is_not_a_run():
    state = _state()
    egress.remember(state, PASSAGE)
    assert egress.overlap(state, "the quarterly revenue of the northern division") == 0


def test_words_the_user_typed_do_not_count():
    state = _state("why did quarterly revenue of the northern division rose by twelve percent")
    egress.remember(state, PASSAGE)
    assert egress.overlap(state, "quarterly revenue of the northern division rose by twelve") == 0


def test_store_is_capped_and_never_in_results():
    state = _state()
    egress.remember(state, " ".join(f"word{i}" for i in range(egress.MAX_HASHES + 100)))
    store = ledger.get_ledger(state)["egress"]
    assert len(store["hashes"]) == egress.MAX_HASHES
    block = json.dumps(TaskManager._security_block(state.context))
    assert "egress" not in block and store["hashes"][0] not in block and "word1" not in block


def test_branches_merge_their_hashes_and_drift_counts():
    root = _state()
    egress.remember(root, "a b c d e f g h")  # creates the salt shared by the branches
    branch = WorkflowState(query=root.query, context=json.loads(json.dumps(root.context)))
    egress.remember(branch, PASSAGE)
    ledger.increment_low_alignment_calls(branch)
    ledger.merge(root, [branch])
    assert egress.overlap(root, "quarterly revenue of the northern division rose by twelve") == 9
    assert ledger.low_alignment_calls(root) == 1

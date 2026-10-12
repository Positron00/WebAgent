"""The committed classifier with its real encoder (the fine-tuned folder in models/triage/encoder/,
or the cached base MiniLM for a frozen classifier). Skipped when either isn't on disk."""
import pytest

from app.services import triage_model

pytestmark = pytest.mark.real_triage_model

if not (triage_model.model_files_present() and triage_model.encoder_available()):
    pytest.skip("triage classifier or MiniLM encoder not available", allow_module_level=True)


@pytest.fixture(scope="module")
def model():
    loaded = triage_model.load_model()
    if loaded is None:
        pytest.skip("triage model failed to load (encoder not cached for offline use?)")
    return loaded


@pytest.mark.parametrize(
    "text,route",
    [
        ("explain what a python list comprehension is", "direct"),
        ("what is the derivative of x squared", "direct"),
        ("fix this TypeError in my javascript function", "direct"),
        ("what are the latest developments in the EU AI Act this month", "agentic"),
        ("compare current pricing of the top three CRM tools for small teams", "agentic"),
        ("bitcoin price now?", "agentic"),
        # Fine-tuning on the first 4x data sent settled history to research; keep it direct.
        ("when was the eiffel tower built", "direct"),
        ("who won the pulitzer prize for fiction this year", "agentic"),
    ],
)
def test_clear_cases(model, text, route):
    p = model.predict_proba(text)
    assert ("agentic" if p >= model.metadata.threshold else "direct") == route


def test_deterministic(model):
    assert model.predict_proba("explain closures") == model.predict_proba("explain closures")


def test_fine_tuned_encoder_is_loaded_from_the_model_folder(model):
    if not model.metadata.encoder_path:
        pytest.skip("committed classifier uses the frozen base encoder")
    assert model.metadata.encoder_path == "encoder"

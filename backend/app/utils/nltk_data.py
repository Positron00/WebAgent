"""Make sure NLTK data is available without making startup depend on the network."""
import logging
import os
from typing import Dict

import nltk

logger = logging.getLogger(__name__)

# Download name -> path that nltk.data.find() looks up.
NLTK_RESOURCES: Dict[str, str] = {
    "punkt": "tokenizers/punkt",
    # NLTK 3.8.2+ sent_tokenize/word_tokenize load punkt_tab, not punkt
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
    "words": "corpora/words",
}


def _is_installed(resource_path: str) -> bool:
    # The downloader leaves some corpora (e.g. wordnet) as a .zip only; NLTK loads those
    # directly, but find("corpora/wordnet") doesn't see them.
    for candidate in (resource_path, f"{resource_path}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_data(*names: str) -> bool:
    """Return True if every named resource is installed.

    Missing resources are downloaded only when WEBAGENT_NLTK_AUTO_DOWNLOAD is true (the default).
    A failed download is logged and never raised, so agents still start offline; features that
    need the missing data fail later with NLTK's own LookupError.
    """
    auto_download = os.getenv("WEBAGENT_NLTK_AUTO_DOWNLOAD", "true").lower() == "true"
    all_present = True
    for name in names:
        if _is_installed(NLTK_RESOURCES.get(name, name)):
            continue
        if auto_download:
            try:
                logger.info(f"Downloading NLTK resource '{name}'")
                if nltk.download(name, quiet=True):
                    continue
            except Exception as e:  # network errors, permission errors
                logger.warning(f"Could not download NLTK resource '{name}': {e}")
        else:
            logger.warning(
                f"NLTK resource '{name}' is not installed and auto-download is off "
                "(run: python -m nltk.downloader " + name + ")"
            )
        all_present = False
    return all_present

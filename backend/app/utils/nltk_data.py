"""Make sure NLTK data is available without making startup depend on the network."""
import logging
import os
from typing import Dict

import nltk

logger = logging.getLogger(__name__)

# Download name -> path that nltk.data.find() looks up.
NLTK_RESOURCES: Dict[str, str] = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
    "words": "corpora/words",
}


def ensure_nltk_data(*names: str) -> bool:
    """Return True if every named resource is installed.

    Missing resources are downloaded only when WEBAGENT_NLTK_AUTO_DOWNLOAD is true (the default).
    A failed download is logged and never raised, so agents still start offline; features that
    need the missing data fail later with NLTK's own LookupError.
    """
    auto_download = os.getenv("WEBAGENT_NLTK_AUTO_DOWNLOAD", "true").lower() == "true"
    all_present = True
    for name in names:
        try:
            nltk.data.find(NLTK_RESOURCES.get(name, name))
            continue
        except LookupError:
            pass
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

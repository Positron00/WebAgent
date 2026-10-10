"""
WebAgent Demo Scripts

This directory contains demonstration scripts for various WebAgent features:

- langgraph_demo.py: Demonstration of the LangGraph multiagent workflow
- multiagent_workflow_demo.py: Comprehensive demonstration of multiagent workflow
- ml_observability_demo.py: Demonstration of ML observability framework
- web_research_demo.py: Demonstration of web research capabilities
- internal_research_demo.py: Demonstration of internal knowledge retrieval
- document_extraction_demo.py: Demonstration of document extraction capabilities
- data_analysis_demo.py: Demonstration of data analysis capabilities
- coding_assistant_demo.py: Demonstration of coding assistant capabilities
- deduplication_demo.py: Demonstration of document deduplication features
- research_results_display.py: Demonstration of research results visualization
- ft_demo.py: Demonstration of fine-tuning capabilities

Run demos with:
python -m backend.demos.[demo_file_name_without_extension]
"""

__description__ = "Demo modules for the WebAgent Platform"
__url__ = "https://github.com/username/webagent"
__license__ = "MIT"
__copyright__ = "Copyright 2025"

import os
import sys

# Add the parent directory to the path so that imports work correctly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import __version__  # noqa: E402  (single source: package.json)
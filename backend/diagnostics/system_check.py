#!/usr/bin/env python
"""
WebAgent System Check
====================

This is a convenience script to run the diagnostics suite.
It forwards to the full diagnostics runner.
"""
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Forward to the diagnostics runner
from backend.diagnostics.core import print_diagnostics_report

def main():
    """Run the system diagnostic checks and print a report."""
    print("\nRunning WebAgent System Check...\n")
    print_diagnostics_report(run_checks=True)
    print("\nSystem check complete.\n")

if __name__ == "__main__":
    main() 
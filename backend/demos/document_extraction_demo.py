#!/usr/bin/env python
"""
Document Extraction Agent Demo
==============================

This demo demonstrates the capabilities of the Document Extraction Agent,
which specializes in extracting structured information from documents.

Usage:
    python -m backend.demos.document_extraction_demo

Requirements:
    - Sample documents in the demo_docs/ directory
    - Configured environment variables (check .env.example)
"""

import os
import sys
import asyncio
import logging
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
import json
from pprint import pprint

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging - only show INFO and higher level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("document_extraction_demo")

# Suppress specific loggers that produce warning messages
logging.getLogger("numexpr").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

# Import agents and services
from backend.app.agents.document_extraction_agent import get_document_extraction_agent
from backend.app.services.document_loader import get_document_loader_service
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.utils.metrics import timing_decorator

# Import PyQt5 for GUI
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextBrowser, QVBoxLayout, QWidget, 
                           QTabWidget, QPushButton, QHBoxLayout, QFileDialog, QMessageBox,
                           QCheckBox, QLabel, QComboBox)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette

# Path to demo documents
DEMO_DOCS_DIR = os.path.join(current_dir, "demo_docs")
os.makedirs(DEMO_DOCS_DIR, exist_ok=True)

# Create a sample document if none exists
def create_sample_document():
    """Create a sample document for testing if none exists."""
    sample_path = os.path.join(DEMO_DOCS_DIR, "sample_financial_report.txt")
    
    if not os.path.exists(sample_path):
        with open(sample_path, "w") as f:
            f.write("""
Financial Report - Q1 2025
==========================

Executive Summary
----------------
In Q1 2025, WebAgent Corp achieved significant growth with revenue reaching $12.5M, 
representing a 15% increase compared to Q1 2024. EBITDA improved by 22% to $3.2M, 
while net profit rose to $1.8M.

Financial Highlights
-------------------
- Revenue: $12.5M (+15% YoY)
- EBITDA: $3.2M (+22% YoY)
- Net Profit: $1.8M (+18% YoY)
- Operating Margin: 25.6% (up from 23.1%)
- Cash Reserves: $8.4M (as of March 31, 2025)

Department Breakdown
-------------------
1. Product Development: $4.2M investment, focused on AI agent capabilities
2. Sales & Marketing: $2.8M, with 65% digital marketing allocation
3. Operations: $1.9M, improved efficiency by 12%
4. Research: $2.1M, primarily for new vector database integration
5. Administration: $1.5M

Market Analysis
--------------
The AI agent market grew by approximately 28% in Q1 2025, with WebAgent gaining 2.3% 
market share. Key competitors include AgentWorks (18.2% share), AI Solutions (15.6%), 
and TechAgents (12.1%).

Outlook for Q2 2025
------------------
We project continued growth in Q2 with revenue estimates between $13.8M-$14.5M and 
EBITDA of $3.5M-$3.8M. Key initiatives include launching the new WebAgent Enterprise 
platform and expanding to three new international markets.
            """)
        logger.info(f"Created sample document: {sample_path}")
    return sample_path


def format_extraction_result(result, dark_mode=False):
    """
    Format extraction result as Markdown
    
    Args:
        result: The extraction result dictionary
        dark_mode: Whether to use dark mode formatting
        
    Returns:
        Formatted Markdown string
    """
    if not result:
        return "No results available."
    
    # Set colors based on dark/light mode
    divider = "* * *" if dark_mode else "---"
    
    # For error cases
    if result.get("error"):
        return f"# Error\n\n{result['error']}"
    
    # Handle LLM extraction result
    extraction_data = result.get("extraction_data", {})
    extraction_type = result.get("extraction_type", "unknown")
    
    markdown = f"# Document Extraction Results\n\n"
    markdown += f"**Extraction Type**: {extraction_type}\n\n"
    markdown += f"{divider}\n\n"
    
    if extraction_type == "llm":
        # Format LLM extraction with improved section formatting
        if "main_topics" in extraction_data:
            markdown += "# Main Topics\n\n"
            for topic in extraction_data.get("main_topics", []):
                markdown += f"• {topic}\n"
            markdown += f"\n{divider}\n\n"
        
        if "key_entities" in extraction_data:
            markdown += "# Key Entities\n\n"
            for entity in extraction_data.get("key_entities", []):
                markdown += f"• {entity}\n"
            markdown += f"\n{divider}\n\n"
        
        if "summary" in extraction_data:
            markdown += "# Summary\n\n"
            markdown += f"{extraction_data.get('summary')}\n\n"
            markdown += f"{divider}\n\n"
        
        if "key_points" in extraction_data:
            markdown += "# Key Points\n\n"
            for i, point in enumerate(extraction_data.get("key_points", []), 1):
                markdown += f"{i}. {point}\n"
            markdown += f"\n{divider}\n\n"
        
        if "document_type" in extraction_data:
            markdown += f"# Document Type\n\n{extraction_data.get('document_type')}\n\n"
            markdown += f"{divider}\n\n"
    
    elif extraction_type == "lda":
        # Format LDA extraction with improved section formatting
        markdown += "# Topics\n\n"
        for topic in result.get("topics", []):
            topic_id = topic.get("topic_id", 0)
            terms = topic.get("terms", [])
            terms_str = ", ".join([f"{term[0]} ({term[1]:.2f})" for term in terms])
            markdown += f"**Topic {topic_id}**: {terms_str}\n\n"
        
        markdown += f"{divider}\n\n"
        
        markdown += "# Important Sentences\n\n"
        for i, sentence in enumerate(result.get("important_sentences", []), 1):
            markdown += f"{i}. {sentence}\n"
        markdown += f"\n{divider}\n\n"
        
        markdown += f"# Analysis Metrics\n\n**Coherence Score**: {result.get('coherence_score', 0):.4f}\n\n"
        markdown += f"{divider}\n\n"
    
    # Add timestamp
    markdown += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    
    return markdown


class ResultsWindow(QMainWindow):
    """
    Window to display extraction results
    """
    def __init__(self, results, document_path):
        super().__init__()
        self.results = results
        self.document_path = document_path
        self.dark_mode = False
        self.setWindowTitle("Document Extraction Results")
        self.setMinimumSize(950, 700)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Add document info
        doc_name = os.path.basename(self.document_path)
        self.info_label = QTextBrowser()
        self.info_label.setMaximumHeight(60)
        self.info_label.setMarkdown(f"## Document: {doc_name}")
        main_layout.addWidget(self.info_label)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs)
        
        # Add results to tabs
        for i, (title, result) in enumerate(self.results):
            tab = QTextBrowser()
            tab.setOpenExternalLinks(True)
            tab.setFont(QFont("Helvetica", 12))
            
            # Set markdown content
            markdown_content = format_extraction_result(result, self.dark_mode)
            tab.setMarkdown(markdown_content)
            
            self.tabs.addTab(tab, title)
        
        # Add display options layout
        options_layout = QHBoxLayout()
        
        # Add theme selection
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.currentIndexChanged.connect(self.toggle_theme)
        
        options_layout.addWidget(theme_label)
        options_layout.addWidget(self.theme_combo)
        options_layout.addStretch()
        
        main_layout.addLayout(options_layout)
        
        # Add buttons layout
        buttons_layout = QHBoxLayout()
        
        # Add save button
        save_btn = QPushButton("Save Results")
        save_btn.setIcon(QIcon.fromTheme("document-save"))
        save_btn.clicked.connect(self.save_results)
        buttons_layout.addWidget(save_btn)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.setIcon(QIcon.fromTheme("window-close"))
        close_btn.clicked.connect(self.close)
        buttons_layout.addWidget(close_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # Apply initial light theme
        self.apply_theme(dark=False)
        
        # Center window
        self.center()
    
    def toggle_theme(self, index):
        """Toggle between light and dark themes"""
        self.dark_mode = (index == 1)  # Dark mode is index 1
        self.apply_theme(dark=self.dark_mode)
        
        # Update content with new theme
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            title = self.tabs.tabText(i)
            for result_title, result in self.results:
                if result_title == title:
                    markdown_content = format_extraction_result(result, self.dark_mode)
                    tab.setMarkdown(markdown_content)
                    break
    
    def apply_theme(self, dark=False):
        """Apply light or dark theme to the application"""
        if dark:
            # Dark theme
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #1e1e1e;
                    color: #e0e0e0;
                }
                QTabWidget::pane {
                    border: 1px solid #4a4a4a;
                    background-color: #2d2d2d;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    background-color: #3a3a3a;
                    color: #b0b0b0;
                    padding: 8px 12px;
                    margin-right: 2px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 1px solid #4a4a4a;
                    border-bottom-color: #2d2d2d;
                }
                QPushButton {
                    background-color: #0c6fc0;
                    color: white;
                    padding: 6px 12px;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1a86e0;
                }
                QPushButton:pressed {
                    background-color: #0a5aa0;
                }
                QTextBrowser {
                    background-color: #2d2d2d;
                    color: #e0e0e0;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    selection-background-color: #0c6fc0;
                    selection-color: white;
                }
                QLabel {
                    color: #e0e0e0;
                }
                QComboBox {
                    background-color: #3a3a3a;
                    color: #e0e0e0;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    padding: 4px;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox QAbstractItemView {
                    background-color: #3a3a3a;
                    color: #e0e0e0;
                    selection-background-color: #0c6fc0;
                }
            """)
        else:
            # Light theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f5f5f5;
                }
                QTabWidget::pane {
                    border: 1px solid #cccccc;
                    background-color: white;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    background-color: #e0e0e0;
                    color: #505050;
                    padding: 8px 12px;
                    margin-right: 2px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background-color: white;
                    color: #2c3e50;
                    border: 1px solid #cccccc;
                    border-bottom-color: white;
                }
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    padding: 6px 12px;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #1c6ea4;
                }
                QTextBrowser {
                    background-color: white;
                    color: #2c3e50;
                    border: 1px solid #e0e0e0;
                    border-radius: 4px;
                    selection-background-color: #3498db;
                    selection-color: white;
                }
                QLabel {
                    color: #2c3e50;
                }
                QComboBox {
                    background-color: white;
                    color: #2c3e50;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 4px;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox QAbstractItemView {
                    background-color: white;
                    color: #2c3e50;
                    selection-background-color: #3498db;
                }
            """)
        
    def center(self):
        """Center window on screen"""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
    
    def save_results(self):
        """Save results to a file"""
        current_tab = self.tabs.currentIndex()
        title = self.tabs.tabText(current_tab)
        
        # Get default filename based on document and extraction type
        doc_name = os.path.splitext(os.path.basename(self.document_path))[0]
        default_name = f"{doc_name}_{title.lower().replace(' ', '_')}_extraction.md"
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", default_name, "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                # Get current tab content
                tab = self.tabs.currentWidget()
                content = tab.toMarkdown()
                
                # Save to file
                with open(file_path, 'w') as f:
                    f.write(content)
                
                QMessageBox.information(self, "Success", f"Results saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")


@timing_decorator
async def run_document_extraction_demo():
    """Run the document extraction agent demo."""
    logger.info("Starting Document Extraction Agent Demo")
    
    # Initialize results list for the GUI
    results = []
    
    # Ask for user confirmation to connect to the TogetherAI LLM
    print("\n" + "=" * 80)
    print("This demo will connect to the TogetherAI LLM API using your configured API key.")
    try:
        model_info = settings.LLM_PROVIDER if hasattr(settings, 'LLM_PROVIDER') else 'together'
        model_name = settings.DEFAULT_MODEL if hasattr(settings, 'DEFAULT_MODEL') else 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'
        print(f"The model that will be used is: {model_name} (Provider: {model_info})")
    except AttributeError:
        print("Using default TogetherAI LLM configuration")
    print("=" * 80)
    
    confirmation = input("\nDo you want to continue and connect to the LLM API? (Y/n): ")
    if confirmation.lower() not in ["y", "yes",""]:
        logger.info("User declined to connect to the LLM API. Demo aborted.")
        print("Demo aborted by user.")
        return
    
    logger.info("User confirmed to connect to the LLM API.")
    print("\nConnecting to TogetherAI LLM API...")
    
    # Create a sample document if needed
    sample_doc_path = create_sample_document()
    
    # Get document loader service
    doc_loader = get_document_loader_service()
    
    # Get document extraction agent
    extraction_agent = get_document_extraction_agent()
    
    # Load document
    logger.info(f"Loading document: {sample_doc_path}")
    try:
        artifact_id = await doc_loader.load_document(sample_doc_path)
        logger.info(f"Document loaded. Artifact ID: {artifact_id}")
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        return
    
    # Create workflow state
    state = WorkflowState(
        query="Extract financial metrics and department budgets from the Q1 2025 financial report",
        inputs={"document_id": artifact_id}
    )
    
    # Run extraction agent
    logger.info("Running document extraction agent...")
    try:
        result_state = await extraction_agent.run(state)
        
        # Print results
        logger.info("Document Extraction Results:")
        if result_state.output:
            pprint(result_state.output)
            # Add to results for GUI
            results.append(("Financial Metrics", result_state.output))
        else:
            logger.warning("No extraction results returned.")
            
    except Exception as e:
        logger.error(f"Error running document extraction agent: {e}")
    
    # Try another extraction with different query
    logger.info("\nRunning another extraction with different query...")
    state.query = "Create a summary of market analysis and outlook from the report"
    
    try:
        result_state = await extraction_agent.run(state)
        
        # Print results
        logger.info("Market Analysis Extraction Results:")
        if result_state.output:
            pprint(result_state.output)
            # Add to results for GUI
            results.append(("Market Analysis", result_state.output))
        else:
            logger.warning("No extraction results returned.")
            
    except Exception as e:
        logger.error(f"Error running document extraction agent: {e}")
    
    logger.info("Document Extraction Agent Demo completed.")
    
    # Show results in GUI
    if results:
        # Create Qt application
        app = QApplication([])
        window = ResultsWindow(results, sample_doc_path)
        window.show()
        app.exec_()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_document_extraction_demo()) 
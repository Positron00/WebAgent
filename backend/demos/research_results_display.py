#!/usr/bin/env python
"""
Research Results Display
=======================

This script displays the results from research queries in a GUI window.
It's designed to show query-answer pairs in a tabbed interface with proper formatting.
Can be used by both internal research and document extraction demos.
"""

import os
import sys
import logging
from datetime import datetime
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("research_display")

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import PyQt5 for GUI
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextBrowser, QVBoxLayout, QWidget, 
                           QTabWidget, QPushButton, QHBoxLayout, QFileDialog, QMessageBox,
                           QCheckBox, QLabel, QComboBox, QShortcut)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QKeySequence

class QueryResult:
    """Represents a query and its result"""
    def __init__(self, query, result=None, error=None):
        self.query = query
        self.result = result
        self.error = error
        self.timestamp = datetime.now()
    
    def has_result(self):
        return self.result is not None
    
    def has_error(self):
        return self.error is not None


def format_result(query_result, dark_mode=False):
    """
    Format query result as Markdown
    
    Args:
        query_result: QueryResult object containing query and result
        dark_mode: Whether to use dark mode formatting
        
    Returns:
        Formatted Markdown string
    """
    # Set colors based on dark/light mode
    divider = "* * *" if dark_mode else "---"
    
    # Start with the query - format it to stand out
    markdown = f"# Query\n\n"
    
    # Format the query with markdown quote block to make it stand out
    query_lines = query_result.query.split('\n')
    if len(query_lines) > 1:
        # For multi-line queries, use a quote block
        markdown += "> " + "\n> ".join(query_lines) + "\n\n"
    else:
        # For single line queries, make them bold and larger
        markdown += f"**{query_result.query}**\n\n"
        
    markdown += f"{divider}\n\n"
    
    # Handle error case
    if query_result.has_error():
        markdown += f"# Error\n\n```\n{query_result.error}\n```\n\n"
        markdown += f"{divider}\n\n"
        markdown += "*No results available due to error*"
        return markdown
    
    # Handle empty result case
    if not query_result.has_result():
        markdown += "# Results\n\n*No results returned*\n\n"
        return markdown
    
    # Format actual results
    result = query_result.result
    
    # Check if result is completely empty
    if result == {} or result is None:
        markdown += "# Results\n\n*No results returned*\n\n"
        return markdown
    
    # Handle web research results
    if isinstance(result, dict) and "analysis" in result:
        if result.get("analysis"):
            markdown += "# Analysis\n\n"
            markdown += f"{result['analysis']}\n\n"
            markdown += f"{divider}\n\n"
        
        if "sources" in result and result["sources"]:
            markdown += "# Sources\n\n"
            for i, source in enumerate(result["sources"], 1):
                title = source.get("title", "Untitled")
                url = source.get("url", "No URL")
                snippet = source.get("content_snippet", "")
                
                markdown += f"### {i}. {title}\n"
                markdown += f"**URL**: [{url}]({url})\n\n"
                if snippet:
                    markdown += f"**Preview**: {snippet}\n\n"
            
            markdown += f"{divider}\n\n"
        elif "sources" in result and not result["sources"]:
            markdown += "# Sources\n\n*No sources found*\n\n"
            markdown += f"{divider}\n\n"
            
        # If empty analysis but we have sources, show a message
        if not result.get("analysis") and result.get("sources"):
            markdown += "# Analysis\n\n*No analysis available. Please review the sources directly.*\n\n"
            markdown += f"{divider}\n\n"
            
        return markdown
    
    # Handle coding assistant output
    if isinstance(result, dict) and ("code" in result or "explanation" in result):
        # Show explanation if available
        if "explanation" in result and result["explanation"]:
            markdown += "# Explanation\n\n"
            markdown += f"{result['explanation']}\n\n"
            markdown += f"{divider}\n\n"
        
        # Show code if available
        if "code" in result and result["code"]:
            markdown += "# Generated Code\n\n"
            
            # Format the code properly with syntax highlighting
            code = result["code"]
            
            # Clean up the code - remove markdown code blocks if present
            if code.startswith("```") and code.endswith("```"):
                # Extract code from markdown code blocks
                lines = code.split("\n")
                if len(lines) > 2:  # At least 3 lines (opening, content, closing)
                    # Remove the first and last lines (opening and closing ```)
                    code = "\n".join(lines[1:-1])
                    
                    # If the first line has language identifier (e.g., "python"), remove it
                    if code.startswith("python"):
                        code = code[6:].strip()
            
            # Add the code block with proper formatting
            markdown += f"```python\n{code}\n```\n\n"
            
            # Add a note about saving
            markdown += f"{divider}\n\n"
            markdown += "*Note: The generated code has been saved to the demo_outputs directory.*\n\n"
        
        return markdown
    
    # Handle internal research agent state
    if hasattr(result, 'reports') and 'internal_research' in result.reports:
        internal_research = result.reports.get('internal_research', {})
        findings = internal_research.get('findings', '')
        documents = internal_research.get('documents', [])
        
        # Clean up findings by removing HTML tags and extra whitespace
        if findings:
            # Remove HTML tags
            findings = re.sub(r'<[^>]+>', '', findings)
            # Fix multiple newlines
            findings = re.sub(r'\n{3,}', '\n\n', findings)
            # Remove redundant "FINDINGS:" prefix if present
            findings = re.sub(r'^FINDINGS:\s*', '', findings)
            
            markdown += "# Research Findings\n\n"
            markdown += f"{findings}\n\n"
            markdown += f"{divider}\n\n"
        
        # Add source documents if available
        if documents:
            # Capture unique sources to show deduplication is working
            unique_sources = {}
            for doc in documents:
                if isinstance(doc, dict):
                    metadata = doc.get('metadata', {})
                    source = metadata.get('source', 'Unknown')
                    score = doc.get('score', 0)
                    
                    # Count occurrences of each source
                    if source in unique_sources:
                        unique_sources[source]['count'] += 1
                        if score > unique_sources[source]['best_score']:
                            unique_sources[source]['best_score'] = score
                    else:
                        unique_sources[source] = {
                            'count': 1,
                            'best_score': score
                        }
            
            # Show source document information with deduplication metrics
            markdown += f"# Source Documents\n\n"
            
            # Try to get deduplication metrics if available
            metrics = internal_research.get('metrics', {})
            dedup_metrics = metrics.get('deduplication', {})
            
            if dedup_metrics:
                docs_before = dedup_metrics.get('documents_before_dedup', 0)
                docs_after = dedup_metrics.get('documents_after_dedup', len(documents))
                sources_before = dedup_metrics.get('sources_before', 0)
                sources_after = dedup_metrics.get('sources_after', len(unique_sources))
                
                markdown += f"Found {len(documents)} relevant documents from {len(unique_sources)} unique sources.\n"
                markdown += f"(Reduced from {docs_before} initial documents through deduplication.)\n\n"
            else:
                markdown += f"Found {len(documents)} relevant documents from {len(unique_sources)} unique sources.\n\n"
            
            # First show a summary of sources and their counts
            markdown += "## Sources Summary\n\n"
            for source, info in sorted(unique_sources.items(), key=lambda x: x[1]['best_score'], reverse=True):
                markdown += f"* **{source}**: {info['count']} document(s), best relevance score: {info['best_score']:.2f}\n"
            
            markdown += f"\n{divider}\n\n"
            
            # Then show details of the top documents
            markdown += "## Top Documents\n\n"
            top_docs = documents[:3]  # Only show top 3
            for i, doc in enumerate(top_docs, 1):
                if isinstance(doc, dict):
                    metadata = doc.get('metadata', {})
                    source = metadata.get('source', 'Unknown')
                    score = doc.get('score', None)
                    
                    markdown += f"### Document {i}\n"
                    markdown += f"**Source**: {source}\n"
                    if score is not None:
                        markdown += f"**Relevance Score**: {score:.2f}\n"
                    
                    # Add a preview of the content
                    content = doc.get('page_content', '')
                    if content:
                        preview = content[:200] + "..." if len(content) > 200 else content
                        markdown += f"\n**Preview**:\n```\n{preview}\n```\n\n"
            
            markdown += f"{divider}\n\n"
    # If result is a dictionary with summary and contexts
    elif isinstance(result, dict):
        if "summary" in result:
            markdown += "# Summary\n\n"
            markdown += f"{result['summary']}\n\n"
            markdown += f"{divider}\n\n"
        
        if "key_points" in result:
            markdown += "# Key Points\n\n"
            for i, point in enumerate(result["key_points"], 1):
                markdown += f"{i}. {point}\n"
            markdown += f"\n{divider}\n\n"
        
        if "contexts" in result:
            markdown += "# Sources\n\n"
            for i, context in enumerate(result["contexts"], 1):
                # Show a preview of the source
                preview = context[:200] + "..." if len(context) > 200 else context
                markdown += f"### Source {i}\n\n```\n{preview}\n```\n\n"
        
        # For document extraction results
        if "extraction_type" in result:
            markdown += f"# Extraction Type\n\n{result['extraction_type']}\n\n"
            markdown += f"{divider}\n\n"
            
            if "extraction_data" in result:
                extraction_data = result["extraction_data"]
                
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
                    markdown += "# Document Summary\n\n"
                    markdown += f"{extraction_data.get('summary')}\n\n"
                    markdown += f"{divider}\n\n"
                
                if "key_points" in extraction_data:
                    markdown += "# Key Points\n\n"
                    for i, point in enumerate(extraction_data.get("key_points", []), 1):
                        markdown += f"{i}. {point}\n"
                    markdown += f"\n{divider}\n\n"
        
        # Handle findings directly
        if "findings" in result:
            findings = result["findings"]
            # Clean up findings
            if isinstance(findings, str):
                # Remove HTML tags
                findings = re.sub(r'<[^>]+>', '', findings)
                # Fix multiple newlines
                findings = re.sub(r'\n{3,}', '\n\n', findings)
                # Remove redundant "FINDINGS:" prefix if present
                findings = re.sub(r'^FINDINGS:\s*', '', findings)
            
            markdown += "# Research Findings\n\n"
            markdown += f"{findings}\n\n"
            markdown += f"{divider}\n\n"
        
        # Handle documents directly
        if "documents" in result:
            documents = result["documents"]
            if documents:
                # Suppress the full source document contents, just show a count
                markdown += f"# Source Documents\n\n"
                
                # Try to get deduplication metrics if available
                metrics = result.get('metrics', {})
                dedup_metrics = metrics.get('deduplication', {}) if isinstance(metrics, dict) else {}
                
                if dedup_metrics and isinstance(dedup_metrics, dict):
                    docs_before = dedup_metrics.get('documents_before_dedup', 0)
                    docs_after = dedup_metrics.get('documents_after_dedup', len(documents))
                    
                    if docs_before > docs_after:
                        markdown += f"Found {len(documents)} relevant documents.\n"
                        markdown += f"(Reduced from {docs_before} initial documents through deduplication.)\n\n"
                    else:
                        markdown += f"Found {len(documents)} relevant documents.\n\n"
                else:
                    markdown += f"Found {len(documents)} relevant documents.\n\n"
                
                # Add a brief summary of the top 3 sources without the full content
                top_docs = documents[:3]  # Only show top 3
                for i, doc in enumerate(top_docs, 1):
                    if isinstance(doc, dict):
                        metadata = doc.get('metadata', {})
                        source = metadata.get('source', 'Unknown')
                        score = doc.get('score', None)
                        
                        markdown += f"**Source {i}**: {source}"
                        if score is not None:
                            markdown += f" (Relevance: {score:.2f})"
                        markdown += "\n\n"
                
                markdown += f"{divider}\n\n"
    else:
        # Just display the result as is
        # Clean up the text if it's a string
        if isinstance(result, str):
            # Remove HTML tags
            result = re.sub(r'<[^>]+>', '', result)
            # Fix multiple newlines
            result = re.sub(r'\n{3,}', '\n\n', result)
        
        markdown += "# Results\n\n"
        markdown += f"{result}\n\n"
    
    # Add timestamp
    markdown += f"\n{divider}\n\n"
    markdown += f"*Generated on {query_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*"
    
    return markdown


class ResultsWindow(QMainWindow):
    """
    Window to display research query results
    """
    def __init__(self, results, title="Research Results"):
        super().__init__()
        self.results = results
        self.dark_mode = False
        self.setWindowTitle(title)
        self.setMinimumSize(1500, 1000)  # Increased from 1050, 800 to have more space for tabs
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)
        
        # Add keyboard shortcuts
        wrap_shortcut = QShortcut(QKeySequence("Ctrl+W"), self)
        wrap_shortcut.activated.connect(self.toggle_wrap_shortcut)
        
        # Add header info
        self.info_label = QTextBrowser()
        self.info_label.setMaximumHeight(70)
        self.info_label.setMarkdown(f"## {self.windowTitle()}\n\nDisplaying {len(self.results)} queries")
        self.info_label.setFrameStyle(0)  # Remove frame for a cleaner look
        main_layout.addWidget(self.info_label)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.tabs)
        
        # Add results to tabs
        for i, result in enumerate(self.results):
            tab = QTextBrowser()
            tab.setOpenExternalLinks(True)
            tab.setFont(QFont("Helvetica", 16))  # Increase font size from 14 to 16
            
            # Set markdown content
            markdown_content = format_result(result, self.dark_mode)
            tab.setMarkdown(markdown_content)
            
            # Default to word wrap for better readability
            tab.setLineWrapMode(QTextBrowser.WidgetWidth)
            tab.setLineWidth(900)  # Wider line width for better readability
            tab.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            
            # Keep good readability for code blocks and tables
            tab.setTabStopWidth(40)  # Set reasonable tab stop width
            
            # Add zoom capabilities
            tab.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard | Qt.LinksAccessibleByMouse)
            tab.zoomIn(2)  # Zoom in by 2 steps to make content larger
            
            # Create shortened tab title from query
            query = result.query
            if len(query) > 30:  # Reduced from 50 to 30 characters for better tab fitting
                query = query[:27] + "..."
            
            self.tabs.addTab(tab, f"Q{i+1}: {query}")
        
        # Make tabs more visible and larger
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMinimumHeight(500)  # Ensure tab content has enough vertical space
        self.tabs.setTabsClosable(False)
        self.tabs.setMovable(True)  # Allow tab reordering
        
        # Auto-adjust tab width based on number of tabs
        self.adjust_tab_width()
        
        # Add display options layout
        options_layout = QHBoxLayout()
        
        # Add word wrap toggle
        self.wrap_checkbox = QCheckBox("Word Wrap")
        self.wrap_checkbox.setChecked(True)  # Default is word wrap enabled
        self.wrap_checkbox.stateChanged.connect(self.toggle_word_wrap)
        self.wrap_checkbox.setToolTip("Toggle word wrap (Ctrl+W)")
        options_layout.addWidget(self.wrap_checkbox)
        
        # Add spacing
        options_layout.addSpacing(20)
        
        # Add theme selection
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.currentIndexChanged.connect(self.toggle_theme)
        self.theme_combo.setMinimumWidth(120)  # Make theme dropdown wider
        self.theme_combo.setMinimumHeight(30)  # Make theme dropdown taller
        
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
        
        # Apply initial dark theme
        self.apply_theme(dark=True)
        self.theme_combo.setCurrentIndex(1)  # Set dropdown to Dark theme
        
        # Center window
        self.center()
    
    def toggle_wrap_shortcut(self):
        """Toggle word wrap when shortcut is pressed"""
        # Toggle the checkbox state
        self.wrap_checkbox.setChecked(not self.wrap_checkbox.isChecked())
    
    def toggle_word_wrap(self, state):
        """Toggle between word wrap and horizontal scrolling for all tabs"""
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if state == Qt.Checked:
                # Enable word wrap
                tab.setLineWrapMode(QTextBrowser.WidgetWidth)
                tab.setLineWidth(900)  # Wider line width for better readability
                tab.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                self.statusBar().showMessage("Word wrap enabled", 2000)
            else:
                # Enable horizontal scrolling
                tab.setLineWrapMode(QTextBrowser.NoWrap)
                tab.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                self.statusBar().showMessage("Horizontal scrolling enabled", 2000)
    
    def toggle_theme(self, index):
        """Toggle between light and dark themes"""
        self.dark_mode = (index == 1)  # Dark mode is index 1
        self.apply_theme(dark=self.dark_mode)
        
        # Update content with new theme
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            result = self.results[i]
            markdown_content = format_result(result, self.dark_mode)
            tab.setMarkdown(markdown_content)
    
    def apply_theme(self, dark=False):
        """Apply light or dark theme to the application"""
        if dark:
            # Dark theme
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #1e1e1e;
                    color: #e0e0e0;
                    font-family: 'Segoe UI', 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
                }
                QTabWidget::pane {
                    border: 1px solid #4a4a4a;
                    background-color: #2d2d2d;
                    border-radius: 6px;
                    padding: 2px;
                }
                QTabBar::tab {
                    background-color: #3a3a3a;
                    color: #b0b0b0;
                    padding: 10px 14px;
                    margin-right: 2px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    font-size: 14px;
                    font-weight: bold;
                    min-width: 120px;
                    max-width: 200px;
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
                    padding: 10px 16px;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                    min-height: 36px;
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
                    border-radius: 6px;
                    selection-background-color: #0c6fc0;
                    selection-color: white;
                    font-size: 16px;
                    line-height: 1.6;
                    padding: 8px;
                }
                QLabel {
                    color: #e0e0e0;
                    font-size: 14px;
                    margin-right: 6px;
                }
                QComboBox {
                    background-color: #3a3a3a;
                    color: #e0e0e0;
                    border: 1px solid #4a4a4a;
                    border-radius: 5px;
                    padding: 6px 12px;
                    font-size: 14px;
                    min-height: 32px;
                }
                QComboBox::drop-down {
                    border: none;
                    padding-right: 10px;
                }
                QComboBox QAbstractItemView {
                    background-color: #3a3a3a;
                    color: #e0e0e0;
                    selection-background-color: #0c6fc0;
                    border-radius: 5px;
                }
                QCheckBox {
                    spacing: 8px;
                    font-size: 14px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 4px;
                    border: 1px solid #6c6c6c;
                }
                QCheckBox::indicator:checked {
                    background-color: #0c6fc0;
                    border: 1px solid #0c6fc0;
                }
            """)
            
            # Apply additional custom styling for markdown content
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                tab.document().setDefaultStyleSheet("""
                    h1 { 
                        color: #3498db; 
                        font-size: 32px; 
                        margin-top: 26px; 
                        margin-bottom: 16px; 
                        font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                        font-weight: 600;
                        letter-spacing: -0.5px;
                    }
                    h2 { 
                        color: #2ecc71; 
                        font-size: 26px; 
                        margin-top: 22px; 
                        margin-bottom: 14px; 
                        font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                        font-weight: 500;
                        letter-spacing: -0.3px;
                    }
                    h3 { 
                        color: #f39c12; 
                        font-size: 22px; 
                        margin-top: 18px; 
                        margin-bottom: 12px; 
                        font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                        font-weight: 500;
                    }
                    p { 
                        line-height: 1.7; 
                        margin-top: 12px; 
                        margin-bottom: 12px; 
                        font-size: 16px; 
                        font-family: 'Segoe UI', 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
                    }
                    code { 
                        background-color: #3a3a3a; 
                        color: #e74c3c; 
                        padding: 3px 6px; 
                        border-radius: 4px; 
                        font-size: 15px;
                        font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
                    }
                    pre { 
                        background-color: #3a3a3a; 
                        padding: 14px 16px; 
                        border-radius: 6px; 
                        margin: 14px 0; 
                        border-left: 3px solid #3498db; 
                        font-size: 15px; 
                        overflow-x: auto; 
                        white-space: pre;
                        font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
                        line-height: 1.5;
                    }
                    table { 
                        border-collapse: collapse; 
                        margin: 18px 0; 
                        min-width: 400px;
                        overflow-x: auto;
                        display: block;
                        border-radius: 4px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
                    }
                    th, td { 
                        padding: 10px 16px; 
                        border: 1px solid #4a4a4a; 
                        text-align: left;
                        font-family: 'Segoe UI', 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
                    }
                    th { 
                        background-color: #3a3a3a;
                        color: #e0e0e0;
                        font-weight: 600;
                        font-size: 15px;
                    }
                    tr:nth-child(even) { 
                        background-color: #333333;
                    }
                    ul, ol { 
                        margin-left: 28px;
                        margin-top: 12px;
                        margin-bottom: 12px;
                    }
                    li { 
                        margin: 8px 0; 
                        font-size: 16px;
                        line-height: 1.6;
                        font-family: 'Segoe UI', 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
                    }
                    a { 
                        color: #3498db; 
                        text-decoration: none;
                        border-bottom: 1px solid rgba(52, 152, 219, 0.5);
                        padding-bottom: 1px;
                    }
                    a:hover {
                        border-bottom: 1px solid #3498db;
                    }
                    blockquote { 
                        background-color: #2a2a2a; 
                        border-left: 4px solid #3498db; 
                        padding: 12px 18px; 
                        margin: 16px 0; 
                        font-size: 16px; 
                        font-style: italic;
                        color: #e0e0e0;
                        line-height: 1.7;
                        border-radius: 0 4px 4px 0;
                    }
                    strong { 
                        font-weight: 600; 
                        font-size: 16px; 
                        color: #e0e0e0;
                    }
                    hr {
                        border: none;
                        height: 1px;
                        background-color: #4a4a4a;
                        margin: 24px 0;
                    }
                """)
        else:
            # Light theme
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #f5f5f5;
                    color: #2c3e50;
                    font-family: 'Segoe UI', 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
                }
                QTabWidget::pane {
                    border: 1px solid #cccccc;
                    background-color: white;
                    border-radius: 6px;
                    padding: 2px;
                }
                QTabBar::tab {
                    background-color: #e0e0e0;
                    color: #505050;
                    padding: 10px 14px;
                    margin-right: 2px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    font-size: 14px;
                    font-weight: bold;
                    min-width: 120px;
                    max-width: 200px;
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
                    padding: 10px 16px;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                    min-height: 36px;
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
                    border-radius: 6px;
                    selection-background-color: #3498db;
                    selection-color: white;
                    font-size: 16px;
                    line-height: 1.6;
                    padding: 8px;
                }
                QLabel {
                    color: #2c3e50;
                    font-size: 14px;
                    margin-right: 6px;
                }
                QComboBox {
                    background-color: white;
                    color: #2c3e50;
                    border: 1px solid #cccccc;
                    border-radius: 5px;
                    padding: 6px 12px;
                    font-size: 14px;
                    min-height: 32px;
                }
                QComboBox::drop-down {
                    border: none;
                    padding-right: 10px;
                }
                QComboBox QAbstractItemView {
                    background-color: white;
                    color: #2c3e50;
                    selection-background-color: #3498db;
                    border-radius: 5px;
                }
                QCheckBox {
                    spacing: 8px;
                    font-size: 14px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 4px;
                    border: 1px solid #bbbbbb;
                }
                QCheckBox::indicator:checked {
                    background-color: #3498db;
                    border: 1px solid #3498db;
                }
            """)
            
            # Apply additional custom styling for markdown content
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                tab.document().setDefaultStyleSheet("""
                    h1 { 
                        color: #2c3e50; 
                        font-size: 32px; 
                        margin-top: 26px; 
                        margin-bottom: 16px; 
                        font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                        font-weight: 600;
                        letter-spacing: -0.5px;
                    }
                    h2 { 
                        color: #16a085; 
                        font-size: 26px; 
                        margin-top: 22px; 
                        margin-bottom: 14px; 
                        font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                        font-weight: 500;
                        letter-spacing: -0.3px;
                    }
                    h3 { 
                        color: #d35400; 
                        font-size: 22px; 
                        margin-top: 18px; 
                        margin-bottom: 12px; 
                        font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                        font-weight: 500;
                    }
                    p { 
                        line-height: 1.7; 
                        margin-top: 12px; 
                        margin-bottom: 12px; 
                        font-size: 16px; 
                        font-family: 'Segoe UI', 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
                    }
                    code { 
                        background-color: #f0f0f0; 
                        color: #e74c3c; 
                        padding: 3px 6px; 
                        border-radius: 4px; 
                        font-size: 15px;
                        font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
                    }
                    pre { 
                        background-color: #f8f8f8; 
                        padding: 14px 16px; 
                        border-radius: 6px; 
                        margin: 14px 0; 
                        border-left: 3px solid #2980b9; 
                        font-size: 15px; 
                        overflow-x: auto; 
                        white-space: pre;
                        font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
                        line-height: 1.5;
                    }
                    table { 
                        border-collapse: collapse; 
                        margin: 18px 0; 
                        min-width: 400px;
                        overflow-x: auto;
                        display: block;
                        border-radius: 4px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }
                    th, td { 
                        padding: 10px 16px; 
                        border: 1px solid #e0e0e0; 
                        text-align: left;
                        font-family: 'Segoe UI', 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
                    }
                    th { 
                        background-color: #f5f5f5;
                        color: #2c3e50;
                        font-weight: 600;
                        font-size: 15px;
                    }
                    tr:nth-child(even) { 
                        background-color: #f9f9f9;
                    }
                    ul, ol { 
                        margin-left: 28px;
                        margin-top: 12px;
                        margin-bottom: 12px;
                    }
                    li { 
                        margin: 8px 0; 
                        font-size: 16px;
                        line-height: 1.6;
                        font-family: 'Segoe UI', 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
                    }
                    a { 
                        color: #2980b9; 
                        text-decoration: none;
                        border-bottom: 1px solid rgba(41, 128, 185, 0.3);
                        padding-bottom: 1px;
                    }
                    a:hover {
                        border-bottom: 1px solid #2980b9;
                    }
                    blockquote { 
                        background-color: #f5f5f5; 
                        border-left: 4px solid #2980b9; 
                        padding: 12px 18px; 
                        margin: 16px 0; 
                        font-size: 16px; 
                        font-style: italic;
                        color: #333333;
                        line-height: 1.7;
                        border-radius: 0 4px 4px 0;
                    }
                    strong { 
                        font-weight: 600; 
                        font-size: 16px; 
                        color: #2c3e50;
                    }
                    hr {
                        border: none;
                        height: 1px;
                        background-color: #e0e0e0;
                        margin: 24px 0;
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
        result = self.results[current_tab]
        
        # Get default filename based on query
        query_text = result.query.lower()
        # Replace special characters and limit length
        safe_query = ''.join(c if c.isalnum() else '_' for c in query_text)
        safe_query = safe_query[:30]  # Limit length
        default_name = f"research_query_{safe_query}_{datetime.now().strftime('%Y%m%d')}.md"
        
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

    def adjust_tab_width(self):
        """Adjust tab width based on the number of tabs"""
        num_tabs = self.tabs.count()
        if num_tabs <= 0:
            return
            
        # Calculate optimal tab width based on window width and number of tabs
        # Leave some margin (100px) for tab bar controls
        available_width = self.width() - 100  
        tab_width = max(120, min(220, available_width // num_tabs))
        
        # Update the tab bar style with the calculated width
        self.tabs.setStyleSheet(f"""
            QTabBar::tab {{
                min-width: {tab_width}px;
                max-width: {tab_width * 2}px;
                min-height: 40px;
                font-size: 14px;
                font-weight: bold;
                padding-left: 12px;
                padding-right: 12px;
                text-align: left;
            }}
            
            QTabWidget::pane {{
                border: 1px solid #4a4a4a;
                border-radius: 6px;
            }}
            
            QTabBar::tab:selected {{
                border-bottom-width: 2px;
            }}
        """)

    def resizeEvent(self, event):
        """Adjust tab widths when window is resized"""
        super().resizeEvent(event)
        self.adjust_tab_width()


def display_research_results(queries, results=None, errors=None, title="Research Results"):
    """
    Display research results in a GUI window
    
    Args:
        queries: List of query strings
        results: List of result dictionaries (can be None for queries with no results)
        errors: List of error messages (can be None for queries with no errors)
        title: Title for the window
    """
    # Validate inputs
    if not queries or not isinstance(queries, list):
        logger.error("No valid queries provided")
        if not queries:
            queries = ["No query provided"]
    
    # Ensure lists are of same length or fill with None
    if results is None:
        results = [None] * len(queries)
    if errors is None:
        errors = [None] * len(queries)
    
    # Validate results - if empty dict, convert to None
    for i in range(len(results)):
        if results[i] == {}:
            results[i] = None
            
    # Create QueryResult objects
    query_results = []
    for i, query in enumerate(queries):
        # If query text is None or empty, provide a default
        if not query:
            query = "No query text provided"
            
        result = results[i] if i < len(results) else None
        error = errors[i] if i < len(errors) else None
        query_results.append(QueryResult(query, result, error))
    
    # Verify we have at least one query to display
    if not query_results:
        logger.error("No queries to display")
        return 1
        
    # Create Qt application
    app = QApplication.instance() or QApplication([])
    window = ResultsWindow(query_results, title=title)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    # Example usage
    queries = [
        "What are our company's AI ethics guidelines?",
        "What is the product roadmap for 2025?",
        "Explain our technical architecture for vector databases",
        "What security measures are in place in our system?"
    ]
    
    # Create dummy errors for demonstration
    errors = [
        "Together received an invalid payload: <ClientResponse(https://api.together.xyz/v1/completions) [404 Not Found]>",
        "Together received an invalid payload: <ClientResponse(https://api.together.xyz/v1/completions) [404 Not Found]>",
        "Together received an invalid payload: <ClientResponse(https://api.together.xyz/v1/completions) [404 Not Found]>",
        "Together received an invalid payload: <ClientResponse(https://api.together.xyz/v1/completions) [404 Not Found]>"
    ]
    
    # Display with example data
    display_research_results(queries, errors=errors) 
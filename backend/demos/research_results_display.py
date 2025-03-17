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
                           QCheckBox, QLabel, QComboBox)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette

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
    
    # Start with the query
    markdown = f"# Query\n\n{query_result.query}\n\n"
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
            # Suppress the full source document contents, just show a count
            markdown += f"# Source Documents\n\n"
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
        self.setMinimumSize(950, 700)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Add header info
        self.info_label = QTextBrowser()
        self.info_label.setMaximumHeight(60)
        self.info_label.setMarkdown(f"## {self.windowTitle()}\n\nDisplaying {len(self.results)} queries")
        main_layout.addWidget(self.info_label)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs)
        
        # Add results to tabs
        for i, result in enumerate(self.results):
            tab = QTextBrowser()
            tab.setOpenExternalLinks(True)
            tab.setFont(QFont("Helvetica", 16))  # Increase font size from 14 to 16
            
            # Set markdown content
            markdown_content = format_result(result, self.dark_mode)
            tab.setMarkdown(markdown_content)
            
            # Improve readability with better text spacing
            tab.setLineWrapMode(QTextBrowser.WidgetWidth)
            tab.setLineWidth(800)  # Wider line width for better readability
            
            # Add zoom capabilities
            tab.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard | Qt.LinksAccessibleByMouse)
            tab.zoomIn(2)  # Zoom in by 2 steps to make content larger
            
            # Create shortened tab title from query
            query = result.query
            if len(query) > 30:
                query = query[:27] + "..."
            
            self.tabs.addTab(tab, f"Query {i+1}: {query}")
        
        # Make tabs more visible and larger
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMinimumHeight(500)  # Ensure tab content has enough vertical space
        self.tabs.setTabsClosable(False)
        self.tabs.setMovable(True)  # Allow tab reordering
        
        # Style tabs with CSS to make them more prominent
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                min-width: 200px;
                min-height: 35px;
                font-size: 13px;
                font-weight: bold;
            }
        """)
        
        # Add display options layout
        options_layout = QHBoxLayout()
        
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
                    font-size: 13px;
                    font-weight: bold;
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
                    font-size: 14px;
                    line-height: 1.5;
                }
                QLabel {
                    color: #e0e0e0;
                    font-size: 14px;
                }
                QComboBox {
                    background-color: #3a3a3a;
                    color: #e0e0e0;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    padding: 4px;
                    font-size: 14px;
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
            
            # Apply additional custom styling for markdown content
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                tab.document().setDefaultStyleSheet("""
                    h1 { color: #3498db; font-size: 34px; margin-top: 20px; margin-bottom: 14px; }
                    h2 { color: #2ecc71; font-size: 28px; margin-top: 16px; margin-bottom: 12px; }
                    h3 { color: #f39c12; font-size: 24px; margin-top: 14px; margin-bottom: 10px; }
                    p { line-height: 1.6; margin-top: 10px; margin-bottom: 10px; font-size: 16px; }
                    code { background-color: #3a3a3a; color: #e74c3c; padding: 2px 4px; border-radius: 3px; font-size: 16px; }
                    pre { background-color: #3a3a3a; padding: 12px; border-radius: 5px; margin: 12px 0; 
                          border-left: 3px solid #3498db; font-size: 16px; }
                    ul, ol { margin-left: 24px; }
                    li { margin: 8px 0; font-size: 16px; }
                    a { color: #3498db; text-decoration: underline; }
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
                    font-size: 13px;
                    font-weight: bold;
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
                    font-size: 14px;
                    line-height: 1.5;
                }
                QLabel {
                    color: #2c3e50;
                    font-size: 14px;
                }
                QComboBox {
                    background-color: white;
                    color: #2c3e50;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 4px;
                    font-size: 14px;
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
            
            # Apply additional custom styling for markdown content
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                tab.document().setDefaultStyleSheet("""
                    h1 { color: #2c3e50; font-size: 34px; margin-top: 20px; margin-bottom: 14px; }
                    h2 { color: #16a085; font-size: 28px; margin-top: 16px; margin-bottom: 12px; }
                    h3 { color: #d35400; font-size: 24px; margin-top: 14px; margin-bottom: 10px; }
                    p { line-height: 1.6; margin-top: 10px; margin-bottom: 10px; font-size: 16px; }
                    code { background-color: #f0f0f0; color: #e74c3c; padding: 2px 4px; border-radius: 3px; font-size: 16px; }
                    pre { background-color: #f8f8f8; padding: 12px; border-radius: 5px; margin: 12px 0; 
                          border-left: 3px solid #2980b9; font-size: 16px; }
                    ul, ol { margin-left: 24px; }
                    li { margin: 8px 0; font-size: 16px; }
                    a { color: #2980b9; text-decoration: underline; }
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


def display_research_results(queries, results=None, errors=None, title="Research Results"):
    """
    Display research results in a GUI window
    
    Args:
        queries: List of query strings
        results: List of result dictionaries (can be None for queries with no results)
        errors: List of error messages (can be None for queries with no errors)
        title: Title for the window
    """
    # Ensure lists are of same length or fill with None
    if results is None:
        results = [None] * len(queries)
    if errors is None:
        errors = [None] * len(queries)
    
    # Create QueryResult objects
    query_results = []
    for i, query in enumerate(queries):
        result = results[i] if i < len(results) else None
        error = errors[i] if i < len(errors) else None
        query_results.append(QueryResult(query, result, error))
    
    # Create Qt application
    app = QApplication([])
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
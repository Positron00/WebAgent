import pytest
import sys
import os
from pathlib import Path

# Try to import path_setup from the backend directory
backend_dir = Path(__file__).resolve().parent.parent.parent
try:
    # First, add the backend directory to the path if it's not already there
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    # Try to import the path_setup module
    from path_setup import setup_path
    setup_path()
except ImportError:
    # Make sure the backend directory is in the path
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

# Try both import styles
try:
    from app.graph import workflows
except ImportError:
    from backend.app.graph import workflows

@pytest.mark.vcr
def test_build_agent_workflow(mocker):
    """Test building the agent workflow."""
    from langchain.schema.runnable import ConfigurableField
    
    # Mock all the required components
    mocker.patch("langchain_core.runnables.passthrough", return_value=mocker.Mock())
    
    # Create a mock StateGraph with nodes attribute
    mock_state_graph = mocker.Mock()
    mock_state_graph.nodes = {"supervisor": "mock_supervisor", "research": "mock_research"}
    mock_state_graph.compile.return_value = mock_state_graph
    
    # Mock StateGraph class
    mock_state_graph_class = mocker.patch("langgraph.graph.StateGraph")
    mock_state_graph_class.return_value = mock_state_graph
    
    # Mock all agent imports
    mocker.patch("app.graph.workflows.get_supervisor_agent", return_value=mocker.Mock())
    mocker.patch("app.graph.workflows.get_document_extraction_agent", return_value=mocker.Mock())
    mocker.patch("app.graph.workflows.get_internal_research_agent", return_value=mocker.Mock())
    mocker.patch("app.graph.workflows.get_web_research_agent", return_value=mocker.Mock())
    mocker.patch("app.graph.workflows.get_senior_research_agent", return_value=mocker.Mock())
    mocker.patch("app.graph.workflows.get_data_analysis_agent", return_value=mocker.Mock())
    mocker.patch("app.graph.workflows.get_coding_assistant_agent", return_value=mocker.Mock())
    mocker.patch("app.graph.workflows.get_team_manager_agent", return_value=mocker.Mock())
    
    # Run the function
    result = workflows.build_agent_workflow({"mock": "config"})
    
    # Check that compile was called
    mock_state_graph.compile.assert_called_once() 
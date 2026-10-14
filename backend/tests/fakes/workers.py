"""Stand-ins for separate uvicorn workers in one test process."""
from app.services.task_manager import TaskManager


def separate_task_manager() -> TaskManager:
    """A TaskManager with its own memory and supervisor, like one in another worker process.

    TaskManager() is a per-process singleton, so this skips it. Every instance uses the same
    results folder (conftest points result_store.RESULTS_DIR at the test's tmp_path).
    """
    manager = object.__new__(TaskManager)
    manager._initialize()
    return manager

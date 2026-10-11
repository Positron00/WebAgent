"""Settings that decide how the app behaves in containers and non-dev environments."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _settings_in_fresh_process(env_overrides: dict, unset: tuple = ()) -> dict:
    """Import app config in a new interpreter (it reads the environment at import time)."""
    code = (
        "import json, sys; sys.path.insert(0, '.');"
        "from app.core.config import settings, BACKEND_DIR;"
        "print(json.dumps({'env': settings.WEBAGENT_ENV, 'api_key_enabled': settings.API_KEY_ENABLED,"
        " 'data_dir': settings.DATA_DIR, 'backend_dir': BACKEND_DIR,"
        " 'rate_limit': settings.RATE_LIMIT_MAX_REQUESTS}))"
    )
    env = {k: v for k, v in {**os.environ, **env_overrides}.items() if k not in unset}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("selected", ["uat", "prod"])
def test_environment_chosen_by_shell_wins_over_env_local(selected):
    # .env.local in a developer checkout sets WEBAGENT_ENV=dev; the shell/container choice must win.
    settings = _settings_in_fresh_process({"WEBAGENT_ENV": selected})
    assert settings["env"] == selected
    assert settings["api_key_enabled"] is True


def test_security_environment_variables_override_yaml():
    # Docs tell operators to set these; the YAML config used to override them silently.
    settings = _settings_in_fresh_process(
        {
            "WEBAGENT_ENV": "dev",
            "WEBAGENT_SECURITY_API_KEY_ENABLED": "true",
            "WEBAGENT_SECURITY_RATE_LIMIT_MAX_REQUESTS": "500",
        }
    )
    assert settings["api_key_enabled"] is True
    assert settings["rate_limit"] == 500


def test_data_paths_are_built_from_the_backend_folder():
    # In the image the backend folder is /app, so data must not be placed under the repo root.
    settings = _settings_in_fresh_process({}, unset=("WEBAGENT_DATA_DIR",))
    assert settings["data_dir"] == os.path.join(settings["backend_dir"], "data")
    assert Path(settings["backend_dir"]) == BACKEND_DIR


def test_version_comes_from_build_arg_when_package_json_is_missing(monkeypatch):
    import app as app_package

    def no_package_json(self, *args, **kwargs):
        raise OSError("not in image")

    monkeypatch.setattr(Path, "read_text", no_package_json)
    monkeypatch.setenv("WEBAGENT_VERSION", "9.9.9")
    assert app_package._read_version() == "9.9.9"


def test_core_transformer_path_resolves_inside_the_image(tmp_path, monkeypatch):
    from backend.models.services import core_transformer_service as svc

    image_root = tmp_path / "app"  # the backend folder, as copied into the image
    monkeypatch.setattr(svc, "BACKEND_DIR", image_root)
    monkeypatch.setattr(svc, "REPO_ROOT", tmp_path / "missing-repo-root")
    service = SimpleNamespace(config=SimpleNamespace(model_path="backend/models/core_transformer"))

    resolved = svc.CoreTransformerModelService._resolve_model_dir(service)

    assert resolved == str(image_root / "models" / "core_transformer")

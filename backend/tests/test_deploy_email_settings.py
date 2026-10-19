"""Topic email can be turned on in a deployment: the backend gets the SMTP and public-URL settings
(recurring-research REC-118, REC-132). They were missing from Compose and Kubernetes until 3.12.1."""
from pathlib import Path

import yaml

BACKEND = Path(__file__).resolve().parents[1]
SETTINGS = ["WEBAGENT_PUBLIC_URL", "WEBAGENT_SMTP_HOST", "WEBAGENT_SMTP_PORT", "WEBAGENT_SMTP_USERNAME",
            "WEBAGENT_SMTP_PASSWORD", "WEBAGENT_SMTP_FROM", "WEBAGENT_SMTP_TLS", "WEBAGENT_SCHEDULER_TOKEN"]


def test_compose_passes_email_settings_to_the_backend():
    compose = yaml.safe_load((BACKEND / "docker-compose.yml").read_text())
    env = compose["services"]["webagent-backend"]["environment"]
    names = {item.split("=", 1)[0] for item in env}
    assert set(SETTINGS) <= names


def test_kubernetes_backend_gets_email_settings_from_the_secret():
    docs = list(yaml.safe_load_all((BACKEND / "kubernetes" / "deployment.yaml").read_text()))
    deployment = next(d for d in docs if d and d.get("kind") == "Deployment")
    backend = next(c for c in deployment["spec"]["template"]["spec"]["containers"] if c["name"] == "webagent-backend")
    env = {e["name"]: e for e in backend["env"]}
    for name in SETTINGS:
        ref = env[name]["valueFrom"]["secretKeyRef"]
        assert ref["name"] == "webagent-secrets" and ref.get("optional") is True


def test_deploy_script_fills_every_secret_key_the_manifest_reads():
    manifest = (BACKEND / "kubernetes" / "deployment.yaml").read_text()
    script = (BACKEND / "kubernetes" / "deploy.sh").read_text()
    keys = {line.split("key:", 1)[1].strip() for line in manifest.splitlines() if line.strip().startswith("key:")}
    for key in keys:
        assert f"--from-literal={key}=" in script, key

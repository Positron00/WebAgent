"""deploy.sh updates Grafana on every deploy (OBS-018, 3.6.0).

Runs the real script with a fake `kubectl` first on PATH: the fake logs each call, prints a minimal
manifest for `create ... --dry-run=client -o yaml` (one data key per `--from-file` file, as kubectl
does) and saves whatever `apply -f -` reads, so the test can check what would reach the cluster.
"""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

BACKEND = Path(__file__).resolve().parents[1]
DEPLOY = BACKEND / "kubernetes" / "deploy.sh"
DASHBOARDS = BACKEND / "grafana" / "dashboards"
ALERTING = BACKEND / "grafana" / "provisioning" / "alerting"

FAKE_KUBECTL = r"""#!/usr/bin/env bash
# Fake kubectl for test_deploy_script.py.
echo "$*" >> "$FAKE_KUBECTL_DIR/calls.log"
case "$1" in
  get)
    # "get namespace X" / "get crd X": exist on an existing cluster, missing on a new one.
    [ "$FAKE_KUBECTL_NEW_CLUSTER" = "1" ] && exit 1
    exit 0
    ;;
  create)
    case " $* " in
      *" --dry-run=client "*) ;;
      *) exit 0 ;;
    esac
    kind="$2"; name="$3"; namespace="default"; files=()
    shift 3
    while [ $# -gt 0 ]; do
      case "$1" in
        -n|--namespace) namespace="$2"; shift ;;
        --from-file=*) files+=("${1#--from-file=}") ;;
      esac
      shift
    done
    [ "$kind" = "configmap" ] && echo "kind: ConfigMap" || echo "kind: Secret"
    echo "apiVersion: v1"
    echo "metadata: {name: $name, namespace: $namespace}"
    echo "data:"
    for source in "${files[@]}"; do
      if [ -d "$source" ]; then
        for f in "$source"/*; do [ -f "$f" ] && echo "  $(basename "$f"): x"; done
      else
        echo "  $(basename "$source"): x"
      fi
    done
    ;;
  apply)
    if [ "$2" = "-f" ] && [ "$3" = "-" ]; then
      count=$(ls "$FAKE_KUBECTL_DIR" | grep -c '^applied-' || true)
      cat > "$FAKE_KUBECTL_DIR/applied-$count.yaml"
    fi
    ;;
esac
exit 0
"""


def _run_deploy(tmp_path, new_cluster):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(FAKE_KUBECTL)
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IXUSR)
    out_dir = tmp_path / "kubectl"
    out_dir.mkdir()
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "FAKE_KUBECTL_DIR": str(out_dir),
        "FAKE_KUBECTL_NEW_CLUSTER": "1" if new_cluster else "0",
        "OPENAI_API_KEY": "test-openai-key",
        "WEBAGENT_SECURITY_SECRET_KEY": "test-secret-key-at-least-32-characters-long",
        "WEBAGENT_API_KEY": "test-api-key",
        "GRAFANA_ADMIN_PASSWORD": "test-grafana-password",
    }
    result = subprocess.run(
        [shutil.which("bash") or "bash", str(DEPLOY), "-e", "uat", "-n", "webagent-uat", "-v", "3.6.0",
         "-r", "reg/webagent-backend", "-H", "host.example"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = (out_dir / "calls.log").read_text().splitlines()
    applied = {}
    for path in sorted(out_dir.glob("applied-*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if doc:
                applied[(doc["kind"], doc["metadata"]["name"])] = doc
    return calls, applied


def _files(folder):
    return {p.name for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")}


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("new_cluster", [False, True], ids=["existing cluster", "first deploy"])
def test_every_deploy_updates_grafana(tmp_path, new_cluster):
    calls, applied = _run_deploy(tmp_path, new_cluster)

    assert "apply -f monitoring/grafana-datasource.yaml" in calls

    dashboards = applied[("ConfigMap", "grafana-dashboard-definitions")]
    assert dashboards["metadata"]["namespace"] == "monitoring"
    assert set(dashboards["data"]) == _files(DASHBOARDS)
    assert "model-drift-dashboard.json" in dashboards["data"]

    alerting = applied[("ConfigMap", "grafana-alerting")]
    assert alerting["metadata"]["namespace"] == "monitoring"
    assert set(alerting["data"]) == _files(ALERTING)
    assert "webagent-alerts.yaml" in alerting["data"]

    restart = "rollout restart deployment/grafana -n monitoring"
    assert restart in calls
    # Grafana restarts after both ConfigMaps are applied (it reads alert rules only at startup).
    configmap_calls = [i for i, c in enumerate(calls) if c.startswith("create configmap grafana-")]
    assert len(configmap_calls) == 2 and calls.index(restart) > max(configmap_calls)


def test_grafana_manifest_mounts_both_configmaps_and_carries_no_dashboard():
    docs = [d for d in yaml.safe_load_all((BACKEND / "kubernetes/monitoring/grafana.yaml").read_text()) if d]
    assert not [d for d in docs if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "grafana-dashboard-definitions"]
    [deployment] = [d for d in docs if d["kind"] == "Deployment"]
    spec = deployment["spec"]["template"]["spec"]
    volumes = {v["name"]: v["configMap"]["name"] for v in spec["volumes"] if "configMap" in v}
    mounts = {m["name"]: m["mountPath"] for m in spec["containers"][0]["volumeMounts"]}
    assert volumes["grafana-dashboard-definitions"] == "grafana-dashboard-definitions"
    assert mounts["grafana-dashboard-definitions"] == "/var/lib/grafana/dashboards"
    assert volumes["grafana-alerting"] == "grafana-alerting"
    assert mounts["grafana-alerting"] == "/etc/grafana/provisioning/alerting"

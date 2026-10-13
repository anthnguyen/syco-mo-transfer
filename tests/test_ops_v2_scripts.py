"""The v2 shell scripts, run for real against stubs (fake python, uv, runpodctl, ssh, curl):
no network, no pod, no money. Run under /bin/bash where it exists (macOS ships bash 3.2).

- run.sh: a SIGTERM mid-pipeline kills the pipeline, does the final HF sync, and only then
  stops the pod (v1 stopped without syncing); a normal run syncs exactly once, then stops.
- stop_pod.sh: the watchdog syncs before it stops; the stop falls back to the REST API.
- launch_pod.sh: a dry run creates nothing and prints no secret; --yes replaces a pod whose
  SSH never came up, bootstraps the run over SSH and confirms it started.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASH = "/bin/bash" if Path("/bin/bash").exists() else shutil.which("bash")
SECRET_HF, SECRET_RP = "hf_FAKESECRET123", "rpa_FAKESECRET456"


def write_exe(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def events(f: Path) -> list[str]:
    return f.read_text().splitlines() if f.exists() else []


def wait_for(pred, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_syntax():
    for f in ("run.sh", "pod.sh", "stop_pod.sh", "smoke.sh", "launch_pod.sh", "pod_status.sh"):
        subprocess.run([BASH, "-n", str(REPO / "scripts" / f)], check=True)


# ------------------------------------------------------------------ run.sh

FAKE_PY = r"""#!/usr/bin/env bash
ev() { echo "$*" >> "$STUB_EVENTS"; }
case "$1" in
  -c) echo "results/$(basename "$3" .yaml)"; exit 0 ;;              # run.sh's out_dir()
  -) cat > /dev/null; exit 1 ;;                                     # smoke_passed_at_head: no
  -u) if [ "$2" = -m ]; then ev "PIPELINE $$"; [ "${STUB_PIPELINE_S:-0}" = 0 ] || exec sleep "$STUB_PIPELINE_S"; exit 0; fi
      [ "$2" = scripts/check_run.py ] && { ev CHECK; exit 0; } ;;
  scripts/preflight.py) ev PREFLIGHT; exit "${STUB_PREFLIGHT_RC:-0}" ;;
  scripts/upload_results.py) case "$*" in *--loop*) ev LOOP; exec sleep 300 ;; *) ev "SYNC $*"; exit 0 ;; esac ;;
esac
ev "UNKNOWN $*"
"""


@pytest.fixture
def run_tree(tmp_path):
    t = tmp_path / "repo"
    (t / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "run.sh", t / "scripts" / "run.sh")
    write_exe(t / "scripts" / "stop_pod.sh", '#!/usr/bin/env bash\necho STOP >> "$STUB_EVENTS"\n')
    write_exe(t / ".venv" / "bin" / "python", FAKE_PY)
    write_exe(t / "bin" / "uv", "#!/usr/bin/env bash\nexit 0\n")
    ev = tmp_path / "events.txt"
    env = dict(os.environ, PATH=f"{t / 'bin'}:{os.environ['PATH']}", STUB_EVENTS=str(ev), SMOKE_FIRST="0",
               RUNPOD_AUTO_STOP="1", RUNPOD_POD_ID="pod-test", SYCOMO_FINAL_SYNC_TIMEOUT="30")
    return t, ev, env


def start_run(t, env, log):
    return subprocess.Popen([BASH, str(t / "scripts" / "run.sh"), "configs/main.yaml"], cwd=t, env=env,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)


def test_run_sigterm_syncs_before_stopping(run_tree, tmp_path):
    t, ev, env = run_tree
    p = start_run(t, dict(env, STUB_PIPELINE_S="300"), tmp_path / "out.log")
    assert wait_for(lambda: any(e.startswith("PIPELINE") for e in events(ev))), (tmp_path / "out.log").read_text()
    pipeline_pid = int(next(e for e in events(ev) if e.startswith("PIPELINE")).split()[1])
    t0 = time.time()
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=60) == 143
    assert time.time() - t0 < 30, "the signal was not handled while the pipeline ran"
    e = [x.split()[0] for x in events(ev)]
    assert e.count("SYNC") == 1 and e.index("SYNC") < e.index("STOP") and e[-1] == "STOP", e
    assert "--timeout 30" in next(x for x in events(ev) if x.startswith("SYNC"))
    with pytest.raises(ProcessLookupError):
        os.kill(pipeline_pid, 0)  # the pipeline was stopped, not orphaned


def test_run_normal_completion(run_tree, tmp_path):
    t, ev, env = run_tree
    p = start_run(t, env, tmp_path / "out.log")
    assert p.wait(timeout=60) == 0, (tmp_path / "out.log").read_text()
    e = [x.split()[0] for x in events(ev) if not x.startswith("LOOP")]
    assert e == ["PREFLIGHT", "PIPELINE", "CHECK", "SYNC", "STOP"], e


def test_run_smoke_first_then_main(run_tree, tmp_path):
    t, ev, env = run_tree
    p = subprocess.Popen([BASH, str(t / "scripts" / "run.sh"), "configs/main.yaml"], cwd=t,
                         env=dict(env, SMOKE_FIRST="1", SYCOMO_SMOKE_CONFIG="configs/v2_smoke.yaml"),
                         stdout=open(tmp_path / "out.log", "w"), stderr=subprocess.STDOUT)
    assert p.wait(timeout=60) == 0, (tmp_path / "out.log").read_text()
    e = [x for x in events(ev) if not x.startswith("LOOP")]
    assert [x.split()[0] for x in e] == ["PREFLIGHT", "PIPELINE", "CHECK", "SYNC"] * 2 + ["STOP"]
    syncs = [x for x in e if x.startswith("SYNC")]
    assert "v2_smoke.yaml" in syncs[0] and "main.yaml" in syncs[1]


def test_run_preflight_failure_stops_without_pipeline(run_tree, tmp_path):
    t, ev, env = run_tree
    p = start_run(t, dict(env, STUB_PREFLIGHT_RC="1"), tmp_path / "out.log")
    assert p.wait(timeout=60) == 1
    assert [x.split()[0] for x in events(ev)] == ["PREFLIGHT", "SYNC", "STOP"]


# ------------------------------------------------------------------ stop_pod.sh

@pytest.fixture
def stop_tree(tmp_path):
    t = tmp_path / "repo"
    (t / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "stop_pod.sh", t / "scripts" / "stop_pod.sh")
    write_exe(t / ".venv" / "bin" / "python", FAKE_PY)
    ev = tmp_path / "events.txt"
    write_exe(t / "bin" / "runpodctl", '#!/usr/bin/env bash\necho "RUNPODCTL $*" >> "$STUB_EVENTS"\nexit "${STUB_RPC_RC:-0}"\n')
    write_exe(t / "bin" / "curl", '#!/usr/bin/env bash\necho "CURL $*" >> "$STUB_EVENTS"\nexit 0\n')
    env = dict(os.environ, PATH=f"{t / 'bin'}:{os.environ['PATH']}", STUB_EVENTS=str(ev), RUNPOD_POD_ID="pod-test",
               RUNPOD_API_KEY=SECRET_RP)
    return t, ev, env


def test_watchdog_syncs_then_stops(stop_tree):
    if subprocess.run(["pgrep", "-f", "bash scripts/run.sh"], capture_output=True).stdout.strip():
        pytest.skip("a real run.sh is running on this machine; the watchdog would signal it")
    t, ev, env = stop_tree
    r = subprocess.run([BASH, "scripts/stop_pod.sh", "--after", "0.0003", "configs/main.yaml"], cwd=t, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    e = events(ev)
    assert e[0].startswith("SYNC") and "--timeout 600" in e[0] and e[1] == "RUNPODCTL pod stop pod-test", e


def test_stop_falls_back_to_rest(stop_tree):
    t, ev, env = stop_tree
    r = subprocess.run([BASH, "scripts/stop_pod.sh"], cwd=t, env=dict(env, STUB_RPC_RC="1"), capture_output=True,
                       text=True, timeout=60)
    e = events(ev)
    assert r.returncode == 0 and "via REST API" in r.stdout
    assert e[:2] == ["RUNPODCTL pod stop pod-test", "RUNPODCTL stop pod pod-test"]
    assert "rest.runpod.io/v1/pods/pod-test/stop" in e[2]


# ------------------------------------------------------------------ launch_pod.sh

FAKE_RUNPODCTL = r"""#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
with open(os.environ["STUB_EVENTS"], "a") as f:
    f.write("RUNPODCTL " + json.dumps(a) + "\n")
out = lambda x: print(json.dumps(x))
if a[:1] == ["version"]:
    print("runpodctl 2.15.0-stub")
elif a[:1] == ["user"]:
    out({"clientBalance": 100.0})
elif a[:2] == ["ssh", "list-keys"]:
    out({"keys": [{"name": "k", "fingerprint": os.environ["STUB_FP"]}]})
elif a[:2] == ["template", "list"]:
    out([{"id": "runpod-torch-v240", "name": "Runpod Pytorch 2.4.0", "category": "NVIDIA", "imageName": "i240"},
         {"id": "runpod-torch-v280", "name": "Runpod Pytorch 2.8.0", "category": "NVIDIA", "imageName": "i280"},
         {"id": "cl", "name": "Runpod Pytorch 2.9.0 for clusters", "category": "NVIDIA", "imageName": "c"}])
elif a[:2] == ["gpu", "list"]:
    out([{"gpuId": "NVIDIA H100 NVL", "available": True, "stockStatus": "Low", "securePricePerHr": 3.19},
         {"gpuId": "NVIDIA H100 80GB HBM3", "available": False, "stockStatus": None, "securePricePerHr": 3.99},
         {"gpuId": "NVIDIA A100 80GB PCIe", "available": True, "stockStatus": "High", "securePricePerHr": 1.79}])
elif a[:2] == ["pod", "create"] and "--help" in a:
    print("--wait  block until ssh is reachable")
elif a[:2] == ["pod", "create"]:
    n = int(open(os.environ["STUB_EVENTS"]).read().count('"create", "--name"'))
    if n == 1:  # the first pod never answers ssh
        sys.stderr.write('waiting for ssh...\n{"error": "wait timeout", "code": "wait_timeout", "id": "stale1"}\n')
        sys.exit(1)
    out({"id": "pod2", "desiredStatus": "RUNNING"})
elif a[:2] == ["ssh", "info"]:
    out({"ip": "10.0.0.1", "port": 2222})
"""

FAKE_SSH = r"""#!/usr/bin/env bash
for cmd; do :; done
echo "SSH $cmd" >> "$STUB_EVENTS"
case "$cmd" in
  *BOOTSTRAP_LAUNCHED*) echo BOOTSTRAP_LAUNCHED ;;
  *RUN_SH_ALIVE*) printf 'code: v2 = abc\nAuto-stop ARMED for pod pod2\nLaunched configs/v2_llama8b.yaml (PID 9)\n---RUN---\nRUN_SH_ALIVE\n[run.sh] uv sync\n' ;;
esac
"""


@pytest.fixture
def launch_env(tmp_path):
    b = tmp_path / "bin"
    write_exe(b / "runpodctl", FAKE_RUNPODCTL)
    write_exe(b / "ssh", FAKE_SSH)
    write_exe(b / "curl", '#!/usr/bin/env bash\necho "CURL $*" >> "$STUB_EVENTS"\necho 200\n')
    key = tmp_path / "key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    fp = subprocess.run(["ssh-keygen", "-lf", f"{key}.pub"], capture_output=True, text=True).stdout.split()[1]
    ev = tmp_path / "events.txt"
    env = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", STUB_EVENTS=str(ev), STUB_FP=fp, HF_TOKEN=SECRET_HF,
               RUNPOD_API_KEY=SECRET_RP, SYCOMO_LAUNCH_POLL_S="0")
    return env, ev, key


def launch(env, key, *args):
    return subprocess.run([BASH, str(REPO / "scripts" / "launch_pod.sh"), "--ssh-key", str(key), "--no-guard", *args],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def test_launch_dry_run_creates_nothing_and_leaks_nothing(launch_env):
    env, ev, key = launch_env
    r = launch(env, key)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "Dry run: nothing created" in out and "PROBLEM" not in out
    assert SECRET_HF not in out and SECRET_RP not in out
    assert "--env '{\"HF_TOKEN\": \"<redacted>\", \"RUNPOD_API_KEY\": \"<redacted>\"" in out
    assert "--template-id runpod-torch-v280" in out and "--gpu-id 'NVIDIA H100 NVL'" in out
    calls = [json.loads(e.split(" ", 1)[1]) for e in events(ev) if e.startswith("RUNPODCTL")]
    assert not [c for c in calls if c[:2] == ["pod", "create"] and "--help" not in c]
    assert not [c for c in calls if c[:2] in (["pod", "delete"], ["pod", "stop"])]
    assert not [e for e in events(ev) if e.startswith("SSH")]


def test_launch_replaces_unreachable_pod_and_starts_run(launch_env):
    env, ev, key = launch_env
    r = launch(env, key, "--yes")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert SECRET_HF not in out and SECRET_RP not in out
    calls = [json.loads(e.split(" ", 1)[1]) for e in events(ev) if e.startswith("RUNPODCTL")]
    creates = [c for c in calls if c[:2] == ["pod", "create"] and "--help" not in c]
    assert len(creates) == 2 and ["pod", "delete", "stale1"] in calls
    c = creates[-1]
    opt = {c[i]: c[i + 1] for i in range(2, len(c) - 1) if c[i].startswith("--") and not c[i + 1].startswith("--")}
    assert opt["--gpu-id"] == "NVIDIA H100 NVL" and opt["--volume-in-gb"] == "150" and opt["--container-disk-in-gb"] == "40"
    assert opt["--ports"] == "22/tcp" and opt["--min-cuda-version"] == "12.8" and opt["--wait-timeout"] == "10m"
    assert "--ssh" in c and "--wait" in c
    pod_env = json.loads(opt["--env"])
    assert pod_env == dict(HF_TOKEN=SECRET_HF, RUNPOD_API_KEY=SECRET_RP, SYCOMO_CONFIG="configs/v2_llama8b.yaml",
                           SYCOMO_MAX_HOURS="16", SYCOMO_REF="v2")
    ssh = [e for e in events(ev) if e.startswith("SSH")]
    assert "curl -fsSL https://raw.githubusercontent.com/anthnguyen/syco-mo-transfer/v2/scripts/pod.sh | bash" in ssh[0]
    assert "setsid" in ssh[0] and "< /dev/null &" in ssh[0]
    assert "Pod pod2 (NVIDIA H100 NVL" in out and "runpodctl pod stop pod2" in out

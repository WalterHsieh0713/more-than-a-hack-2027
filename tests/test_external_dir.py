import os
import subprocess
import sys

from src.match import ROOT

ASK = "from src.external import idsse; print(idsse.cache_path('J03WMX'))"


def cache_path(env_value):
    env = {k: v for k, v in os.environ.items() if k != "MTAH_EXTERNAL_DIR"}
    if env_value:
        env["MTAH_EXTERNAL_DIR"] = env_value
    return subprocess.run([sys.executable, "-c", ASK], capture_output=True, text=True, cwd=ROOT,
                          env=env, check=True).stdout.strip()


def test_open_data_defaults_to_the_repo_folder():
    assert cache_path(None) == os.path.join(ROOT, "external_data", "idsse", "cache", "J03WMX.npz")


def test_the_environment_can_move_open_data_elsewhere():
    assert cache_path("/mnt/ssd/open") == "/mnt/ssd/open/idsse/cache/J03WMX.npz"

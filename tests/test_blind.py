import os
import re
import subprocess
import sys

import pytest

from src.match import ROOT
from tests.blind_fixture import SECRET


def blind(tmp_path, *names):
    env = dict(os.environ, MTAH_DATA_DIR=str(tmp_path))
    return subprocess.run([sys.executable, "-m", "src.blind", "--registry", "tests.blind_fixture",
                           *names], capture_output=True, text=True, cwd=ROOT, env=env)


@pytest.mark.parametrize("name", ["passes", "raises", "asserts", "exits_early", "dies"])
def test_no_value_ever_reaches_the_terminal(tmp_path, name):
    done = blind(tmp_path, name)
    assert SECRET not in done.stdout + done.stderr
    assert done.stderr == ""


def test_a_passing_check_reports_only_pass(tmp_path):
    done = blind(tmp_path, "passes")
    assert done.stdout == "PASS passes\n" and done.returncode == 0


@pytest.mark.parametrize("name,error", [("raises", "ValueError"), ("asserts", "AssertionError"),
                                        ("exits_early", "SystemExit")])
def test_a_failing_check_reports_only_type_and_location(tmp_path, name, error):
    done = blind(tmp_path, name)
    assert re.fullmatch(r"FAIL {}: {} at blind_fixture\.py:\d+\n".format(name, error), done.stdout)
    assert done.returncode == 1


def test_dying_without_a_verdict_is_a_failure_even_with_exit_code_zero(tmp_path):
    done = blind(tmp_path, "dies")
    assert done.stdout == "FAIL dies: crashed\n" and done.returncode == 1


def test_the_full_output_is_kept_locally_for_a_person(tmp_path):
    blind(tmp_path, "raises")
    logs = [f for f in os.listdir(tmp_path / "logs") if f.endswith(".log")]
    assert len(logs) == 1
    assert SECRET in (tmp_path / "logs" / logs[0]).read_text()


def test_one_failure_fails_the_run_but_every_check_still_reports(tmp_path):
    done = blind(tmp_path, "raises", "passes")
    assert done.returncode == 1
    assert "PASS passes\n" in done.stdout and "FAIL raises" in done.stdout


def test_an_unknown_check_runs_nothing(tmp_path):
    done = blind(tmp_path, "passes", "nonexistent")
    assert done.returncode == 2
    assert not (tmp_path / "logs").exists()

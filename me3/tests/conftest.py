import os
import shutil
import tempfile
from pathlib import Path

import yaml

# Point every server at a throwaway data dir before any server module is imported.
_tmp = Path(tempfile.mkdtemp(prefix="me3-test-"))
_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))
_cfg["paths"] = {k: str(_tmp / k) for k in _cfg["paths"]}
(_tmp / "config.yaml").write_text(yaml.safe_dump(_cfg))
os.environ["ME3_CONFIG"] = str(_tmp / "config.yaml")


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_tmp, ignore_errors=True)

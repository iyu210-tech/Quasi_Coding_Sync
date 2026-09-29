"Fixture regeneration"

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fibonacci_qc.regression import fixture_paths, write_regression_fixture, assert_matches_regression_fixture

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("pass --confirm for deliberate independent regeneration")
    targets = fixture_paths()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = targets[0].parent/"backups"/stamp
    backup.mkdir(parents=True, exist_ok=False)
    for path in targets:
        if path.exists():
            shutil.copy2(path, backup/path.name)
    output = write_regression_fixture(confirm=True)
    assert_matches_regression_fixture()
    print("Regenerated:", output)
    print("Previous fixture backup:", backup)

if __name__ == "__main__":
    main()
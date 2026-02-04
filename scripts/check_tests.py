import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    test_file = repo_root / "tests" / "test_env.py"

    if not test_file.exists():
        print(f"Missing test file: {test_file}")
        return 2

    cmd = [sys.executable, "-m", "pytest", str(test_file)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

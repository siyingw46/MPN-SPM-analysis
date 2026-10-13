"""Compare the active Python environment with the observed run02 package inventory.

This reads package metadata only. It does not install packages or run analyses.
"""
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

EXPECTED_PYTHON = (3, 12, 10)
PACKAGE_LIST = Path(__file__).parent / 'environment' / 'requirements-observed-run02.txt'


def expected_packages():
    pins = {}
    for raw in PACKAGE_LIST.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        name, version = line.split('==', 1)
        pins[name] = version
    return pins


def main():
    rows = []
    for name, expected in expected_packages().items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        rows.append({
            'package': name,
            'observed_run02': expected,
            'installed': actual,
            'status': 'MATCH' if actual == expected else ('MISSING' if actual is None else 'DIFFERENT'),
        })
    python_match = sys.version_info[:3] == EXPECTED_PYTHON
    passed = python_match and all(row['status'] == 'MATCH' for row in rows)
    print(json.dumps({
        'status': 'PASS' if passed else 'MISMATCH',
        'python': sys.version,
        'expected_python': '.'.join(map(str, EXPECTED_PYTHON)),
        'platform': platform.platform(),
        'packages': rows,
        'note': 'Observed local-run environment; no installation or analysis was performed.',
    }, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()

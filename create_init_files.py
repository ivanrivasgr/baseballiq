#!/usr/bin/env python3
"""
create_init_files.py
====================
Run once during project setup to create all __init__.py files.
    python create_init_files.py
"""

from pathlib import Path

PACKAGES = [
    "pipeline",
    "pipeline/ingestion",
    "pipeline/silver",
    "pipeline/gold",
    "enrichment",
    "models",
    "reports",
    "dashboard",
    "dashboard/pages",
    "dashboard/components",
    "tests",
]

for pkg in PACKAGES:
    path = Path(pkg) / "__init__.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text('"""{}"""\n'.format(pkg.replace("/", ".")))
        print(f"Created: {path}")
    else:
        print(f"Exists:  {path}")

print("\nAll __init__.py files ready.")

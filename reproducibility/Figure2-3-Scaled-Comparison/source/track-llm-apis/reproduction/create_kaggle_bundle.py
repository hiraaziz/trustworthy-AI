"""Package the current reproduction source for upload as a Kaggle Dataset."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "track-llm-apis"
OUTPUT = REPO_ROOT / "b3it-kaggle-source.zip"
INCLUDE = ("pyproject.toml", "uv.lock", "src", "reproduction")
EXCLUDE_PARTS = {"__pycache__", ".venv", ".pytest_cache"}


def main() -> None:
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        for entry_name in INCLUDE:
            entry = PACKAGE_ROOT / entry_name
            entries = [entry] if entry.is_file() else sorted(entry.rglob("*"))
            for path in entries:
                if not path.is_file() or EXCLUDE_PARTS.intersection(path.parts):
                    continue
                archive.write(path, Path("track-llm-apis") / path.relative_to(PACKAGE_ROOT))
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()

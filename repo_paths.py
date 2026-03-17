from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def resolve_repo_file(filename: str, required: bool = True) -> str:
    path = REPO_ROOT / filename
    if required and not path.exists():
        raise FileNotFoundError(
            f"Required repository file not found: {path}\n"
            f"Please make sure you are running inside a complete copy of the FHMM repository."
        )
    return str(path)

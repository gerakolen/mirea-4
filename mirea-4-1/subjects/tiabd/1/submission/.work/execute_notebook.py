import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: execute_notebook.py PATH")

    path = Path(sys.argv[1]).resolve()
    notebook = nbformat.read(path, as_version=4)
    os.environ["JUPYTER_PATH"] = str(
        Path(__file__).resolve().parents[1]
        / "subjects"
        / "tiabd"
        / "1"
        / ".venv"
        / "share"
        / "jupyter"
    )
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="tiabd-pr1",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"executed: {path}")


if __name__ == "__main__":
    main()

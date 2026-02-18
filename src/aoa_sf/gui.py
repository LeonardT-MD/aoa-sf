from __future__ import annotations
import sys
from pathlib import Path


def main():
    try:
        import streamlit.web.cli as stcli
    except ImportError as e:
        raise SystemExit(
            "GUI dependencies not installed.\n"
            "Install with:\n"
            "  pip install -e '.[ui]'\n"
        ) from e

    app_path = Path(__file__).with_name("streamlit_app.py")

    # Mimic: streamlit run <app_path>
    sys.argv = ["streamlit", "run", str(app_path)]
    stcli.main()


if __name__ == "__main__":
    main()

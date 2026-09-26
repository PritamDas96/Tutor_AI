"""
Windows-safe launcher for the GenAI-Tutor Streamlit apps.

Why this exists
---------------
On Windows, Streamlit executes the app script in a background "script runner"
thread. PyTorch's `c10.dll` sometimes fails to initialize when it is first
imported from a non-main thread, raising:

    OSError: [WinError 1114] A dynamic link library (DLL) initialization
             routine failed. Error loading ...torch\\lib\\c10.dll

The RAG / agentic editions import `sentence_transformers` (→ torch) at module
top level, so they hit this the moment a browser connects. Importing torch in
the MAIN thread here — before Streamlit starts — means the worker-thread import
is just a cached no-op, and the error disappears.

(This is only needed on some Windows setups; Linux/Streamlit Cloud is unaffected.)

Usage
-----
    python run_app.py <app_file> [port]

Examples
--------
    python run_app.py TUtor_AI.py 8501
    python run_app.py Tutor_AI_RAG.py 8502
    python run_app.py Tutor_AI-RAG-Langsmith.py 8503
    python run_app.py Tutor_AI_AGENTIC.py 8504
"""
import sys

# --- main-thread preload (the actual fix) ---
try:
    import torch  # noqa: F401
    import sentence_transformers  # noqa: F401
except Exception:
    # v1 (TUtor_AI.py) doesn't need torch; ignore if not installed.
    pass

from streamlit.web import cli as stcli

if __name__ == "__main__":
    app = sys.argv[1] if len(sys.argv) > 1 else "Tutor_AI_AGENTIC.py"
    port = sys.argv[2] if len(sys.argv) > 2 else "8501"
    sys.argv = [
        "streamlit", "run", app,
        "--server.port", port,
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
    ]
    sys.exit(stcli.main())

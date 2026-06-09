"""Launch the MindForm Console -- a browser cockpit for the personality engine.

    python console.py                 # serve on http://127.0.0.1:8000/
    PORT=9000 python console.py       # choose a port
    python console.py --port 9000     # ditto

Pick or create a character, then talk to it: every message is an experience that
nudges its five OCEAN traits, and you watch them move live. This is a thin UI
layer -- all personality math stays in the engine modules, untouched.
"""

from web.server import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Catan Arena Server

Runs Flask server with both Catanatron and Arena APIs.

Usage:
    python -m catan_arena.web.server
    # or
    python catan_arena/web/server.py
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Flask, send_from_directory
from flask_cors import CORS


def create_app():
    """Create Flask app with all blueprints."""
    static_folder = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_folder))
    CORS(app)  # Enable CORS for frontend

    # Register Arena API (LLM support)
    from catan_arena.web.arena_api import bp as arena_bp
    app.register_blueprint(arena_bp)

    # Try to register original Catanatron API (optional - needs SQLAlchemy)
    try:
        from catanatron.web.api import bp as catanatron_bp
        app.register_blueprint(catanatron_bp)
        print("  ✓ Original Catanatron API loaded")
    except ImportError as e:
        print(f"  ⚠ Original Catanatron API not available: {e}")

    # Health check
    @app.route("/health")
    def health():
        return {"status": "ok", "service": "catan-arena"}

    # Serve Arena UI
    @app.route("/arena")
    def arena_ui():
        return send_from_directory(app.static_folder, "arena.html")

    # API info
    @app.route("/")
    def index():
        return {
            "service": "Catan Arena API",
            "endpoints": {
                "catanatron": "/api/games (original Catanatron API)",
                "arena": "/api/arena/games (Arena API with LLM support)",
                "health": "/health",
            },
            "docs": "See README for API documentation",
        }

    return app


def main():
    """Run the server."""
    app = create_app()

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    CATAN ARENA SERVER                        ║
╠══════════════════════════════════════════════════════════════╣
║  Server running at: http://{host}:{port}                       ║
║                                                              ║
║  Endpoints:                                                  ║
║    - Original API: http://localhost:{port}/api/games          ║
║    - Arena API:    http://localhost:{port}/api/arena/games    ║
║                                                              ║
║  To create an LLM game:                                      ║
║    POST /api/arena/games                                     ║
║    {{"players": ["LLM:Claude", "LLM:GPT4", "RANDOM"]}}         ║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

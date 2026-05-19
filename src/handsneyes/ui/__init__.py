"""Command Center: web UI + REST API for monitoring and driving the agent.

Watches the homer's run directories for new frames, exposes them over HTTP,
and runs ControllerAgent intents on demand. See server.py for the FastAPI
app and static/ for the mobile-first web UI.
"""

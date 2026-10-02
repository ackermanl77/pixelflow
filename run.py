import uvicorn

from backend.config import HOST, port

# The port comes from backend/config.py (8288, not ComfyUI's 8188;
# PIXELFLOW_PORT overrides it) - the same place desktop.py reads it from.

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=HOST, port=port(), reload=True)

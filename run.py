import uvicorn

# Not ComfyUI's default 8188 (see desktop.py); keep the two in sync.

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8288, reload=True)

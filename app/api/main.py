from fastapi import FastAPI

app = FastAPI(title="Whatdog API", version="2.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}
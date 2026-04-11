from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.router import router

app = FastAPI(title="SurgiPath Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "SurgiPath Backend (FastAPI + WebRTC) is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.infrastructure.routes import router
from app.core.config import settings

app = FastAPI(
    title="BETO Model Service",
    version="1.0.0",
    description="Servicio de inferencia del modelo BETO en GPU"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router)
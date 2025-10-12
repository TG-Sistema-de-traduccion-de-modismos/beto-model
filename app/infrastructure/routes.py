from fastapi import APIRouter, HTTPException
from app.application.analyzer import BETOModismosAnalyzer
from app.domain.models import PredictRequest, PredictResponse
from app.core.logging_config import logger
import torch

router = APIRouter(prefix="", tags=["BETO Model"])

analyzer = BETOModismosAnalyzer()


@router.on_event("startup")
def startup_event():
    logger.info("🚀 Iniciando BETO Model Service (GPU)...")
    analyzer.load_model()
    logger.info(f"Modelo cargado: {analyzer.is_loaded}")
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ GPU no disponible, usando CPU")


@router.get("/health")
def health_check():
    logger.info("🩺 Health check solicitado en modelo.")
    return {
        "status": "healthy",
        "model_loaded": analyzer.is_loaded,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    logger.info(f"Solicitud de predicción recibida: {request.texto[:50]}...")
    
    if not analyzer.is_loaded:
        logger.error("Modelo no está cargado")
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    if not request.texto.strip():
        logger.warning("Texto vacío recibido")
        raise HTTPException(status_code=400, detail="El campo 'texto' es obligatorio")
    
    try:
        resultado = analyzer.analizar_texto(request.texto)
        logger.info(f"Predicción exitosa: {len(resultado.get('modismos_detectados', {}))} modismos detectados")
        
        return PredictResponse(
            status="success",
            resultado=resultado,
            error=None
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
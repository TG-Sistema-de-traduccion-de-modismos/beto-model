import json
import re
import time
import torch
import unicodedata
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from app.domain.vocabulario import VOCABULARIO
from app.core.logging_config import logger
from app.core.config import settings


def quitar_tildes(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


class BETOModismosAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.is_loaded = False

    def load_model(self):
        try:
            logger.info(f"Cargando modelo desde Hugging Face: {settings.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)

            self.setup_label_encoder()
            self.is_loaded = True
            logger.info("Modelo BETO-Finetuned cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}")
            self.is_loaded = False

    def setup_label_encoder(self):
        dataset_path = Path(settings.dataset_path)
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            all_labels = [et for entrada in data for et in entrada.get("etiquetas", [])]
            unique_labels = sorted(set(all_labels))

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(unique_labels)
            logger.info(f"LabelEncoder configurado con {len(unique_labels)} clases.")
        except FileNotFoundError:
            logger.warning("dataset.json no encontrado, usando clases por defecto.")
            fallback_classes = [
                'comida', 'lenguaje', 'malo', 'nada', 'ordinario', 'salir',
                'sorpresa', 'trabajo', 'tramite', 'preminente', 'diligencias',
                'mostrar', 'complicado', 'afinar', 'inacabado', 'acechando',
                'cotidiano', 'flojo'
            ]
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(sorted(fallback_classes))

    def detectar_palabras(self, texto: str):
        texto_norm = quitar_tildes(texto.lower())
        encontradas = []
        for palabra, raiz in VOCABULARIO.items():
            raiz_norm = quitar_tildes(raiz.lower())
            if re.search(rf"\b{re.escape(raiz_norm)}[a-záéíóúüñ]*\b", texto_norm, re.IGNORECASE):
                encontradas.append(palabra)
        return encontradas

    def marcar_palabra_objetivo(self, texto: str, palabra: str):
        raiz = VOCABULARIO.get(palabra)
        if not raiz:
            return texto
        raiz_norm = quitar_tildes(raiz.lower())
        return re.sub(
            rf"\b{re.escape(raiz_norm)}[a-záéíóúüñ]*\b",
            lambda m: f"[TGT] {m.group(0)} [TGT]",
            texto.lower(),
            count=1,
            flags=re.IGNORECASE
        )

    def predecir_significado(self, texto_marcado: str):
        if not self.is_loaded:
            raise RuntimeError("El modelo no está cargado.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self.tokenizer(texto_marcado, return_tensors="pt", truncation=True,
                                padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        return self.label_encoder.inverse_transform([pred])[0]

    def analizar_texto(self, texto: str):
        palabras = self.detectar_palabras(texto)
        if not palabras:
            return {
                "texto_original": texto,
                "modismos_detectados": {},
                "modismos_detallados": [],
                "total_modismos": 0,
                "timestamp": time.time(),
                "modelo": "BETO-Finetuned"
            }

        analizados = {}
        detalles = []
        for palabra in palabras:
            try:
                marcado = self.marcar_palabra_objetivo(texto, palabra)
                significado = self.predecir_significado(marcado)
                analizados[palabra] = significado
                detalles.append({
                    "palabra": palabra,
                    "significado_detectado": significado,
                    "contexto": texto,
                    "confianza": "alta" if significado != "desconocido" else "baja"
                })
            except Exception as e:
                logger.error(f"Error analizando '{palabra}': {e}")
                analizados[palabra] = "error"

        return {
            "texto_original": texto,
            "modismos_detectados": analizados,
            "modismos_detallados": detalles,
            "total_modismos": len(analizados),
            "timestamp": time.time(),
            "modelo": "BETO-Finetuned"
        }

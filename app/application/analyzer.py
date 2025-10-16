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
import torch.nn.functional as F


def quitar_tildes(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


class BETOModismosAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.is_loaded = False
        self.device = None

    def load_model(self):
        try:
            logger.info(f"Cargando modelo desde Hugging Face: {settings.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            self.setup_label_encoder()
            self.is_loaded = True
            logger.info("Modelo BETO-Finetuned cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando el modelo: {type(e).__name__}: {e}")
            self.is_loaded = False

    def setup_label_encoder(self):
        dataset_path = Path(settings.dataset_path)
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            data = [d for d in raw_data if isinstance(d, dict) and "etiquetas" in d]
            if not data:
                raise ValueError("El dataset no contiene etiquetas válidas.")

            all_labels = [et for entrada in data for et in entrada.get("etiquetas", [])]
            unique_labels = sorted(set(all_labels))

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(unique_labels)
            logger.info(f"LabelEncoder configurado con {len(unique_labels)} clases.")

            # Imprimir todas las clases
            logger.info("Clases detectadas por el modelo:")
            for i, clase in enumerate(self.label_encoder.classes_, start=1):
                logger.info(f"{i}: {clase}")

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

            # Imprimir clases por defecto
            logger.info("Clases por defecto configuradas:")
            for i, clase in enumerate(self.label_encoder.classes_, start=1):
                logger.info(f"{i}: {clase}")

        except Exception as e:
            logger.error(f"Error configurando LabelEncoder: {type(e).__name__}: {e}")


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

        inputs = self.tokenizer(
            texto_marcado,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            confianza, pred = torch.max(probs, dim=1)

        etiqueta = self.label_encoder.inverse_transform([pred.item()])[0]
        return etiqueta, float(confianza.item())

    def analizar_texto(self, texto: str):
        start_time = time.time()
        palabras = self.detectar_palabras(texto)
        analizados = {}
        detalles = []

        if not palabras:
            logger.info("No se detectaron modismos en el texto.")
            return {
                "texto_original": texto,
                "modismos_detectados": {},
                "modismos_detallados": [],
                "total_modismos": 0,
                "timestamp": start_time,
                "tiempo_procesamiento": 0.0,
                "modelo": "BETO-Finetuned"
            }

        for palabra in palabras:
            try:
                marcado = self.marcar_palabra_objetivo(texto, palabra)
                significado, confianza = self.predecir_significado(marcado)
                analizados[palabra] = significado
                detalles.append({
                    "palabra": palabra,
                    "significado_detectado": significado,
                    "contexto": texto,
                    "confianza": str(round(confianza, 3))
                })
            except Exception as e:
                logger.error(f"Error analizando '{palabra}': {type(e).__name__}: {e}")
                analizados[palabra] = f"error: {type(e).__name__}"

        tiempo_total = round(time.time() - start_time, 3)
        logger.info(f"Análisis completado en {tiempo_total}s. Modismos detectados: {len(analizados)}")

        return {
            "texto_original": texto,
            "modismos_detectados": analizados,
            "modismos_detallados": detalles,
            "total_modismos": len(analizados),
            "timestamp": start_time,
            "tiempo_procesamiento": tiempo_total,
            "modelo": "BETO-Finetuned"
        }


# ---------------------- BLOQUE PARA IMPRIMIR LAS CLASES ----------------------
if __name__ == "__main__":
    analyzer = BETOModismosAnalyzer()
    analyzer.load_model()  # carga el modelo y configura LabelEncoder

    # Registrar todas las clases en los logs
    if analyzer.label_encoder:
        logger.info("Clases detectadas por el modelo:")
        for i, clase in enumerate(analyzer.label_encoder.classes_):
            logger.info(f"{i + 1}: {clase}")

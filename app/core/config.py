from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "pescobarg/BETO-finetuned-modismos"
    dataset_path: str = "dataset/dataset.json"
    app_name: str = "BETO Model Service"
    app_version: str = "1.0.0"
    app_description: str = "Servicio de inferencia del modelo BETO en GPU"
    port: int = 8002

    class Config:
        env_file = ".env"


settings = Settings()
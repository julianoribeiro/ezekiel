"""Configurações centrais do projeto via variáveis de ambiente."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Simulação
    sim_runs: int = 30
    sim_seed: int = 42

    # DCOP/ADOPT — pesos da função de custo f_ij (Equação 4.2 da tese)
    dcop_max_cycles: int = 200
    dcop_gamma1: float = 1.0   # peso Waiting Time
    dcop_gamma2: float = 0.5   # peso Lead Time
    dcop_gamma3: float = 0.3   # peso Throughput (negativo no custo)

    # Output
    output_dir: str = "./output"


settings = Settings()

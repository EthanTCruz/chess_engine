import os


from pydantic_settings import BaseSettings


class Settings(BaseSettings, case_sensitive=True):
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_score_db: int = 1
    redis_mate_db: int = 2

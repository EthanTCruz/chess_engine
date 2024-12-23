from sqlalchemy import create_engine

from chess_engine.src.model.classes.sqlite.models import Base
from chess_engine.src.model.config.config import settings
from sqlalchemy.orm import sessionmaker





engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

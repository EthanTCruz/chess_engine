from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from chess_engine.src.model.classes.sqlite.models import Base
from chess_engine.src.model.config.config import settings
from sqlalchemy.orm import sessionmaker





engine = create_engine(settings.database_url, 
                       connect_args={"check_same_thread": False, "timeout": 30},
                        poolclass=QueuePool,
                        pool_size=20,  
                        max_overflow=10)

SessionLocal = sessionmaker(autocommit=False, 
                                autoflush=False, 
                                bind=engine)


Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

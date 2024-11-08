from chess_engine.src.model.classes.sqlite.models import (GamePositions,
                                                          GamePositionRollup,
                                                          TrainGamePositions,
                                                          TestGamePositions,
                                                          ValidationGamePositions)
from chess_engine.src.model.classes.sqlite.database import SessionLocal
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import get_all_bitboards_dict
from chess_engine.src.model.config.config import Settings
from sqlalchemy.orm import Session, get_db
from typing import List, Tuple
from sqlalchemy import or_, and_, func
import chess
import re
import pandas as pd
import numpy as np
import json 
from tqdm import tqdm

def gpr_generator(game_posiition_rollup_model,yield_size: int,db: Session = next(get_db())):
    try:
        # Querying all records in GamePositions
        for game_position in db.query(GamePositions).yield_per(yield_size):
            yield game_position
    finally:
        db.close()
        yield None


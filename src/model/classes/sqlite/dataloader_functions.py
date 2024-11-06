from chess_engine.src.model.classes.sqlite.models import (GamePositions,
                                                          GamePositionRollup,
                                                          TrainGamePositions,
                                                          TestGamePositions,
                                                          ValidationGamePositions)
from chess_engine.src.model.classes.sqlite.database import SessionLocal
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import get_all_bitboards_dict
from chess_engine.src.model.config.config import Settings
from sqlalchemy.orm import Session
from typing import List, Tuple
from sqlalchemy import or_, and_, func
import chess
import re
import pandas as pd
import numpy as np
import json 
from tqdm import tqdm